# --- main.py: FastAPI Workout Analyzer with Async Vector Search, Polymorphic Data, and OpenAI LLM ---
import json
import numpy as np
from PIL import Image
import io
import base64
import matplotlib
import os
import uvicorn
import asyncio
import logging
import time
import random  # NEW for polymorphic additions
from datetime import datetime, timezone  # NEW for using a real datetime field

from typing import (
    Optional, List, Mapping, Any, Dict, Union, Tuple, ClassVar
)

# --- FastAPI & MongoDB (Async) ---
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
# REMOVED: StaticFiles import is no longer needed
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.operations import SearchIndexModel
from pymongo.errors import OperationFailure, DuplicateKeyError, AutoReconnect
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorCollection,
    AsyncIOMotorCursor
)
from pymongo.results import (
    InsertOneResult,
    InsertManyResult,
    UpdateResult,
    DeleteResult
)
from dotenv import load_dotenv

# --- New Import for OpenAI API call ---
import httpx  # For asynchronous HTTP requests

# --- Part 1: Initial Setup (Logging, Matplotlib, Env) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

matplotlib.use('Agg')
import matplotlib.pyplot as plt

load_dotenv()

# ############################################################################
# ############################################################################
# Part 2: Asynchronous MongoDB Proxy Wrapper (Non-Scoped)
# (Includes AsyncAtlasIndexManager and Proxy classes)
# ############################################################################
# ############################################################################

# --- ROBUST IMPORT FIX ---
try:
    from pymongo import GEO2DSPHERE
except ImportError:
    logger.warning("Could not import GEO2DSPHERE from pymongo. Defining manually.")
    GEO2DSPHERE = "2dsphere"
# --- END FIX ---


class AsyncAtlasIndexManager:
    """
    Manages MongoDB Atlas Search indexes (Vector & Lucene) and standard
    database indexes with an asynchronous (Motor-native) interface.
    """
    __slots__ = ('_collection',)

    DEFAULT_POLL_INTERVAL: ClassVar[int] = 5  # seconds
    DEFAULT_SEARCH_TIMEOUT: ClassVar[int] = 600  # 10 minutes
    DEFAULT_DROP_TIMEOUT: ClassVar[int] = 300   # 5 minutes

    def __init__(self, real_collection: AsyncIOMotorCollection):
        if not isinstance(real_collection, AsyncIOMotorCollection):
            raise TypeError(
                f"Expected AsyncIOMotorCollection, got {type(real_collection)}"
            )
        self._collection = real_collection

    async def create_search_index(
        self,
        name: str,
        definition: Dict[str, Any],
        index_type: str = "search",
        wait_for_ready: bool = True,
        timeout: int = DEFAULT_SEARCH_TIMEOUT
    ) -> bool:
        """
        Creates or updates an Atlas Search index (Robust Version).
        """
        # Ensure collection exists
        try:
            coll_name = self._collection.name
            all_collections = await self._collection.database.list_collection_names()
            if coll_name not in all_collections:
                await self._collection.database.create_collection(coll_name)
                logger.info(f"Created collection '{coll_name}' as it did not exist.")
        except Exception as e:
            logger.error(f"Failed to ensure collection '{self._collection.name}' exists: {e}")
            raise Exception(f"Failed to create prerequisite collection '{self._collection.name}': {e}")

        try:
            # Check for existing index
            existing_index = await self.get_search_index(name)

            if existing_index:
                logger.info(f"Search index '{name}' already exists.")
                latest_def = existing_index.get("latestDefinition", {})
                definition_changed = False
                change_reason = ""

                if "fields" in definition and index_type.lower() == "vectorsearch":
                    existing_fields = latest_def.get("fields")
                    if existing_fields != definition["fields"]:
                        definition_changed = True
                        change_reason = "vector 'fields' definition differs."
                elif "mappings" in definition and index_type.lower() == "search":
                    existing_mappings = latest_def.get("mappings")
                    if existing_mappings != definition["mappings"]:
                        definition_changed = True
                        change_reason = "Lucene 'mappings' definition differs."
                else:
                    logger.warning(
                        f"Index definition '{name}' has keys that don't match "
                        f"index_type '{index_type}'. Cannot reliably check for changes."
                    )

                if definition_changed:
                    logger.warning(f"Search index '{name}' definition has changed ({change_reason}). Triggering update...")
                    await self.update_search_index(
                        name=name,
                        definition=definition,
                        wait_for_ready=False
                    )

                elif existing_index.get("queryable"):
                    logger.info(f"Search index '{name}' is already queryable and definition is up-to-date.")
                    return True
                elif existing_index.get("status") == "FAILED":
                    logger.error(
                        f"Search index '{name}' exists but is in a FAILED state. "
                        f"Manual intervention in Atlas UI may be required."
                    )
                    return False
                else:
                    logger.info(
                        f"Search index '{name}' exists and is up-to-date, "
                        f"but not queryable (Status: {existing_index.get('status')}). Waiting..."
                    )

            else:
                # Create new index
                try:
                    logger.info(f"Creating new search index '{name}' of type '{index_type}'...")
                    search_index_model = SearchIndexModel(
                        definition=definition,
                        name=name,
                        type=index_type
                    )
                    await self._collection.create_search_index(model=search_index_model)
                    logger.info(f"Search index '{name}' build has been submitted.")
                except OperationFailure as e:
                    if "IndexAlreadyExists" in str(e) or "DuplicateIndexName" in str(e):
                        logger.warning(f"Race condition: Index '{name}' was created by another process.")
                    else:
                        logger.error(f"OperationFailure during search index creation for '{name}': {e.details}")
                        raise e

            if wait_for_ready:
                return await self._wait_for_search_index_ready(name, timeout)
            return True

        except OperationFailure as e:
            logger.error(f"OperationFailure during search index creation/check for '{name}': {e.details}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred regarding search index '{name}': {e}")
            raise

    async def get_search_index(self, name: str) -> Optional[Dict[str, Any]]:
        try:
            pipeline = [{"$listSearchIndexes": {"name": name}}]
            async for index_info in self._collection.aggregate(pipeline):
                return index_info
            return None
        except OperationFailure as e:
            logger.error(f"OperationFailure retrieving search index '{name}': {e.details}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving search index '{name}': {e}")
            return None

    async def list_search_indexes(self) -> List[Dict[str, Any]]:
        try:
            return await self._collection.list_search_indexes().to_list(None)
        except Exception as e:
            logger.error(f"Error listing search indexes: {e}")
            return []

    async def drop_search_index(
        self,
        name: str,
        wait_for_drop: bool = True,
        timeout: int = DEFAULT_DROP_TIMEOUT
    ) -> bool:
        try:
            if not await self.get_search_index(name):
                logger.info(f"Search index '{name}' does not exist. Nothing to drop.")
                return True

            await self._collection.drop_search_index(name=name)
            logger.info(f"Submitted request to drop search index '{name}'.")

            if wait_for_drop:
                return await self._wait_for_search_index_drop(name, timeout)
            return True
        except OperationFailure as e:
            if "IndexNotFound" in str(e):
                logger.info(f"Search index '{name}' was already deleted (race condition).")
                return True
            logger.error(f"OperationFailure dropping search index '{name}': {e.details}")
            raise
        except Exception as e:
            logger.error(f"Error dropping search index '{name}': {e}")
            raise

    async def update_search_index(
        self,
        name: str,
        definition: Dict[str, Any],
        wait_for_ready: bool = True,
        timeout: int = DEFAULT_SEARCH_TIMEOUT
    ) -> bool:
        try:
            logger.info(f"Updating search index '{name}'...")
            await self._collection.update_search_index(name=name, definition=definition)
            logger.info(f"Search index '{name}' update submitted. Rebuild initiated.")
            if wait_for_ready:
                return await self._wait_for_search_index_ready(name, timeout)
            return True
        except OperationFailure as e:
            logger.error(f"Error updating search index '{name}': {e.details}")
            raise
        except Exception as e:
            logger.error(f"Error updating search index '{name}': {e}")
            raise

    async def _wait_for_search_index_ready(self, name: str, timeout: int) -> bool:
        start_time = time.time()
        logger.info(f"Waiting up to {timeout}s for search index '{name}' to become queryable...")

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.error(f"Timeout: Index '{name}' did not become queryable within {timeout}s.")
                raise TimeoutError(f"Index '{name}' did not become queryable within {timeout}s.")

            index_info = None
            try:
                index_info = await self.get_search_index(name)
            except (OperationFailure, AutoReconnect) as e:
                logger.warning(f"DB Error during polling for index '{name}': {getattr(e, 'details', e)}. Retrying...")
            except Exception as e:
                logger.error(f"Unexpected error during polling for index '{name}': {e}. Retrying...")

            if index_info:
                status = index_info.get("status")
                if status == "FAILED":
                    logger.error(f"Search index '{name}' failed to build (Status: FAILED). Check Atlas UI for details.")
                    raise Exception(f"Index build failed for '{name}'.")

                queryable = index_info.get("queryable")
                if queryable:
                    logger.info(f"Search index '{name}' is queryable (Status: {status}).")
                    return True

                logger.info(f"Polling for '{name}'. Status: {status}. Queryable: {queryable}. Elapsed: {elapsed:.0f}s")
            else:
                logger.info(f"Polling for '{name}'. Index not found yet (normal during creation). Elapsed: {elapsed:.0f}s")

            await asyncio.sleep(self.DEFAULT_POLL_INTERVAL)

    async def _wait_for_search_index_drop(self, name: str, timeout: int) -> bool:
        start_time = time.time()
        logger.info(f"Waiting up to {timeout}s for search index '{name}' to be dropped...")
        while True:
            if time.time() - start_time > timeout:
                logger.error(f"Timeout: Index '{name}' was not dropped within {timeout}s.")
                raise TimeoutError(f"Index '{name}' was not dropped within {timeout}s.")

            index_info = await self.get_search_index(name)
            if not index_info:
                logger.info(f"Search index '{name}' has been successfully dropped.")
                return True

            logger.debug(f"Polling for '{name}' drop. Still present. Elapsed: {time.time() - start_time:.0f}s")
            await asyncio.sleep(self.DEFAULT_POLL_INTERVAL)

    # --- Regular Database Index Methods ---
    async def create_index(
        self,
        keys: Union[str, List[Tuple[str, Union[int, str]]]],
        **kwargs: Any
    ) -> str:
        if isinstance(keys, str):
            keys = [(keys, ASCENDING)]

        index_name = kwargs.get("name")
        if not index_name:
            try:
                from pymongo.helpers import _index_list
                index_doc = MongoClient()._database._CommandBuilder._gen_index_doc(keys, kwargs)
                index_name = _index_list(index_doc['key'].items())
            except Exception:
                index_name = f"index_{'_'.join([k[0] for k in keys])}"
                logger.warning(f"Could not auto-generate index name, using fallback: {index_name}")

        try:
            existing_indexes = await self.list_indexes()
            for index in existing_indexes:
                if index.get("name") == index_name:
                    logger.info(f"Regular index '{index_name}' already exists.")
                    return index_name

            name = await self._collection.create_index(keys, **kwargs)
            logger.info(f"Successfully created regular index '{name}'.")
            return name
        except OperationFailure as e:
            logger.error(f"OperationFailure creating regular index '{index_name}': {e.details}")
            raise
        except Exception as e:
            logger.error(f"Failed to create regular index '{index_name}': {e}")
            raise

    async def create_text_index(
        self, fields: List[str], weights: Optional[Dict[str, int]] = None,
        name: str = "text_index", **kwargs: Any
    ) -> str:
        keys = [(field, TEXT) for field in fields]
        if weights:
            kwargs["weights"] = weights
        if name:
            kwargs["name"] = name
        return await self.create_index(keys, **kwargs)

    async def create_geo_index(
        self, field: str,
        name: Optional[str] = None, **kwargs: Any
    ) -> str:
        keys = [(field, GEO2DSPHERE)]
        if name:
            kwargs["name"] = name
        return await self.create_index(keys, **kwargs)

    async def drop_index(self, name: str):
        try:
            await self._collection.drop_index(name)
            logger.info(f"Successfully dropped regular index '{name}'.")
        except OperationFailure as e:
            if "index not found" in str(e).lower():
                logger.info(f"Regular index '{name}' does not exist. Nothing to drop.")
            else:
                logger.error(f"Failed to drop regular index '{name}': {e.details}")
                raise
        except Exception as e:
            logger.error(f"Failed to drop regular index '{name}': {e}")
            raise

    async def list_indexes(self) -> List[Dict[str, Any]]:
        try:
            return await self._collection.list_indexes().to_list(None)
        except Exception as e:
            logger.error(f"Error listing regular indexes: {e}")
            return []

    async def get_index(self, name: str) -> Optional[Dict[str, Any]]:
        indexes = await self.list_indexes()
        return next((index for index in indexes if index.get("name") == name), None)


class CollectionProxy:
    """Wraps an AsyncIOMotorCollection for index management access and basic proxying."""
    __slots__ = ('_collection', '_index_manager')

    def __init__(self, real_collection: AsyncIOMotorCollection):
        self._collection = real_collection
        self._index_manager: Optional[AsyncAtlasIndexManager] = None

    @property
    def index_manager(self) -> AsyncAtlasIndexManager:
        if self._index_manager is None:
            self._index_manager = AsyncAtlasIndexManager(self._collection)
        return self._index_manager

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return getattr(self._collection, name)


class DbProxy:
    """Wraps an AsyncIOMotorDatabase to provide non-scoped collection access."""
    __slots__ = ('_db', '_wrapper_cache')

    def __init__(self, real_db: AsyncIOMotorDatabase):
        self._db = real_db
        self._wrapper_cache: Dict[str, CollectionProxy] = {}

    def __getattr__(self, name: str) -> Union[CollectionProxy, Any]:
        if name in self._wrapper_cache:
            return self._wrapper_cache[name]

        real_attr = getattr(self._db, name)
        if isinstance(real_attr, AsyncIOMotorCollection):
            wrapper = CollectionProxy(real_collection=real_attr)
            self._wrapper_cache[name] = wrapper
            return wrapper
        else:
            return real_attr


MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable not set.")

async_client = AsyncIOMotorClient(MONGO_URI)
real_db = async_client["workout_db"]
db = DbProxy(real_db=real_db)

VECTOR_INDEX_NAME = "workout_vector_index"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-3.5-turbo"

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. LLM features will be disabled until set.")

PLACEHOLDER_SUMMARY = "Click 'Generate AI Summary' to analyze this workout using OpenAI and MongoDB Vector Search context."
PLACEHOLDER_CLASSIFICATION = "Pending Analysis"
PLACEHOLDER_PROMPT = "Prompt context is generated on-the-fly and displayed here when the workout detail page loads, even before the summary is generated."

VECTOR_INDEX_DEF = {
    "fields": [
        {
            "type": "vector",
            "path": "workout_vector",
            "numDimensions": 192,
            "similarity": "cosine"
        },
        {
            "type": "filter",
            "path": "_id"
        }
    ]
}

# --- Part 4: Data Generation Logic (Augmented for Polymorphic Data) ---
def create_synthetic_apple_watch_data(workout_id_suffix: int = 0):
    """
    Creates a polymorphic workout document with random variations:
    - Outdoor Run, Strength Training, Cycling, or Yoga
    - Always includes a 64-length time_series for HR, Calories, Speed
    - Adds extra fields like gear_used, session_tag, etc.
    """
    np.random.seed(workout_id_suffix)  # deterministic per ID
    t = np.linspace(0, 2 * np.pi, 64)
    hr_base = 110 + (workout_id_suffix % 7) * 5
    cal_base = 6 + (workout_id_suffix % 5) * 1
    speed_base = 4.0 + (workout_id_suffix % 6) * 0.5

    hr_pattern = hr_base + 60 * np.sin(t - np.pi / 2 + np.random.rand() * 0.5) + np.random.rand(64) * 10
    cal_pattern = cal_base + 4 * np.sin(t - np.pi / 2 + np.random.rand() * 0.5) + np.random.rand(64) * 2

    if workout_id_suffix % 4 == 0:
        speed_pattern = speed_base * (
            np.sin(t * 4 + np.random.rand() * 0.5) > 0.5
        ) + np.random.rand(64) * 0.5
    elif workout_id_suffix % 4 == 1:
        speed_pattern = 3.0 + t * (speed_base / (2 * np.pi)) + np.random.rand(64) * 0.3
    else:
        speed_pattern = np.full(64, speed_base * 0.8) + np.random.rand(64) * 0.5

    speed_pattern[:5] = 2.0 + np.random.rand(5) * 0.5
    speed_pattern[-5:] = 1.0 + np.random.rand(5) * 0.5
    hr_pattern[:5] -= 20
    hr_pattern[-5:] -= 10

    hr_pattern = np.maximum(hr_pattern, 50)
    cal_pattern = np.maximum(cal_pattern, 0)
    speed_pattern = np.maximum(speed_pattern, 0)

    doc = {
        "_id": f"workout_6b421a9c_{workout_id_suffix}",
        "user_id": f"user_{789 + workout_id_suffix % 10}",
        "start_time": datetime(2025, 10, 27, 10, (10 + workout_id_suffix % 40), 0, tzinfo=timezone.utc),
        "duration_minutes": 64,
        "workout_type": "Outdoor Run",
        "time_series": {
            "heart_rate": list(np.round(hr_pattern, 2)),
            "calories_per_min": list(np.round(cal_pattern, 2)),
            "speed_kph": list(np.round(speed_pattern, 2))
        }
    }

    # Polymorphic logic
    type_idx = workout_id_suffix % 4
    if type_idx == 0:
        doc["workout_type"] = "Strength Training"
        doc["sets_reps"] = [
            {"exercise": "squat", "reps": 10, "weight_kg": 60},
            {"exercise": "bench", "reps": 8, "weight_kg": 70}
        ]
        doc["rpe"] = int(np.random.randint(5, 10))
    elif type_idx == 1:
        # keep as Outdoor Run
        pass
    elif type_idx == 2:
        doc["workout_type"] = "Cycling"
        doc["cadence_rpm"] = list(np.round(np.random.rand(64) * 80 + 70, 2))
        doc["elevation_gain_m"] = int(np.random.randint(50, 501))
    else:
        doc["workout_type"] = "Yoga"
        doc["focus_area"] = np.random.choice(["Hips & Mobility", "Full Body Flow", "Restorative", "Power Yoga"])
        doc["mood_rating"] = int(np.random.randint(1, 6))

    doc["gear_used"] = [
        {"item": "Shoes v3", "kilometers": float(np.random.randint(50, 201))},
        {"item": "HRM Strap", "battery_life_percent": int(np.random.randint(10, 101))}
    ]

    doc["session_tag"] = np.random.choice(["Race Day", "Recovery", "Z2 Cardio", "Tempo Pace", "Threshold"])

    doc["post_session_notes"] = {
        "hydration_ml": int(np.random.randint(500, 2501)),
        "notes": "Felt good, slight fatigue" if np.random.rand() > 0.5 else "Pushed harder than usual"
    }

    return doc


def normalize_data(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    clipped_data = np.clip(data, min_val, max_val)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(clipped_data, dtype=np.uint8)
    normalized = (clipped_data - min_val) / range_val
    return (normalized * 255).astype(np.uint8)


NORM_BOUNDS = {
    "heart_rate": (50, 200),
    "calories_per_min": (0, 20),
    "speed_kph": (0, 15)
}

def generate_workout_viz_arrays(doc: dict, image_dim: int = 8) -> dict:
    required_length = image_dim * image_dim
    default_1d = np.zeros(required_length)
    default_2d = np.zeros((image_dim, image_dim), dtype=np.uint8)
    default_rgb = np.zeros((image_dim, image_dim, 3), dtype=np.uint8)
    error_result = {
        "raw_hr": default_1d, "raw_cal": default_1d, "raw_speed": default_1d,
        "channel_r_2d": default_2d, "channel_g_2d": default_2d, "channel_b_2d": default_2d,
        "rgb_combined": default_rgb
    }

    try:
        ts = doc['time_series']
        hr_data = np.array(ts['heart_rate'])
        cal_data = np.array(ts['calories_per_min'])
        speed_data = np.array(ts['speed_kph'])

        if not (len(hr_data) == required_length and len(cal_data) == required_length and len(speed_data) == required_length):
            raise ValueError(f"Time series data must have {required_length} elements.")

        r_norm = normalize_data(hr_data, *NORM_BOUNDS["heart_rate"])
        g_norm = normalize_data(cal_data, *NORM_BOUNDS["calories_per_min"])
        b_norm = normalize_data(speed_data, *NORM_BOUNDS["speed_kph"])

        r_2d = r_norm.reshape(image_dim, image_dim)
        g_2d = g_norm.reshape(image_dim, image_dim)
        b_2d = b_norm.reshape(image_dim, image_dim)

        return {
            "raw_hr": hr_data, "raw_cal": cal_data, "raw_speed": speed_data,
            "channel_r_2d": r_2d, "channel_g_2d": g_2d, "channel_b_2d": b_2d,
            "rgb_combined": np.stack([r_2d, g_2d, b_2d], axis=-1)
        }
    except (KeyError, TypeError, ValueError) as e:
        logger.error(f"Error processing time series for doc '{doc.get('_id', 'N/A')}': {e}")
        return error_result


def encode_pil_image_to_base64(
    img_array: np.ndarray,
    resize_dim: Optional[Tuple[int, int]] = None,
    mode: str = 'RGB',
    color: Optional[Tuple[int, int, int]] = None
) -> str:
    error_placeholder = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42m"
                       "NkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
    try:
        if mode == 'L' and color and img_array.ndim == 2:
            colored_array = np.zeros((*img_array.shape, 3), dtype=np.uint8)
            for i in range(3):
                colored_array[..., i] = 255
                if color[i] == 0:
                    colored_array[..., i] = 255 - img_array
                elif color[i] < 255:
                    colored_array[..., i] = 255 - ((255 - color[i]) * img_array // 255)
            img = Image.fromarray(colored_array, 'RGB')
        elif img_array.ndim == 3 and img_array.shape[2] == 3 and mode == 'RGB':
            img = Image.fromarray(img_array, 'RGB')
        elif img_array.ndim == 2 and mode == 'L':
            img = Image.fromarray(img_array, 'L')
        else:
            raise ValueError(f"Unsupported array shape/mode: {img_array.shape}, mode='{mode}'")

        if resize_dim:
            img = img.resize(resize_dim, Image.NEAREST)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return error_placeholder


def generate_chart_base64(data: np.ndarray, title: str, color: str) -> str:
    # --- Matplotlib styling for dark mode ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig.patch.set_facecolor('#132A38') # Atlas dark blue
    ax.set_facecolor('#132A38')
    
    ax.plot(data, color=color, linewidth=2)
    ax.set_title(title, fontsize=10, color='#F9FAFB')
    ax.set_xlim(0, len(data) - 1 if len(data) > 1 else 1)
    
    # Hide axes and ticks
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#23435B')
    ax.spines['left'].set_color('#23435B')

    plt.tight_layout()
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="PNG", facecolor=fig.get_facecolor())
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating chart '{title}': {e}")
        return ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAA"
                "AAYAAjCB0C8AAAAASUVORK5CYII=")
    finally:
        plt.close(fig)
        plt.style.use('default') # Reset style


def get_feature_vector(doc: dict) -> np.ndarray:
    arrays = generate_workout_viz_arrays(doc, 8)
    return arrays['rgb_combined'].flatten()


app = FastAPI()
generation_lock = asyncio.Lock()

# REMOVED: app.mount for /static is no longer needed


# --- LLM Functions ---
async def call_openai_api(prompt: str) -> str:
    system_prompt = (
        "You are a professional Workout Radiologist. Analyze the provided structured workout "
        "data and provide a concise, qualitative summary (maximum 3 sentences) from the "
        "perspective of a fitness expert. Focus only on the pattern and function of the effort."
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            logger.info(f"Calling OpenAI API with model {OPENAI_MODEL}...")
            response = await client.post(OPENAI_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            summary = data['choices'][0]['message']['content'].strip()
            logger.info("OpenAI API call successful.")
            return summary
        except httpx.HTTPStatusError as e:
            error_msg = f"OpenAI API HTTP error: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=f"LLM API Error (HTTP {e.response.status_code}): Check API key, quota, or service status."
            )
        except Exception as e:
            error_msg = f"An unexpected error occurred during OpenAI API call: {e}"
            logger.error(error_msg, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"LLM API Error: Unexpected exception: {e.__class__.__name__}"
            )


def analyze_time_series_features(doc: dict, nearest_neighbors: List[Dict[str, Any]]) -> Tuple[str, str]:
    ts = doc['time_series']
    hr_data = np.array(ts['heart_rate'])
    cal_data = np.array(ts['calories_per_min'])
    speed_data = np.array(ts['speed_kph'])

    hr_avg = np.mean(hr_data)
    hr_max = np.max(hr_data)
    speed_std = np.std(speed_data)
    cal_total = np.sum(cal_data)

    if speed_std > 2.5 and hr_max > 180:
        classification = "High-Intensity Interval Training"
        visual_cue = "Speed and HR channels show high-contrast, jagged peaks (Horizontal Stripes)."
    elif speed_std < 0.5 and hr_avg > 130:
        classification = "Steady State Aerobic Run"
        visual_cue = "All channels show low contrast and smooth, uniform color."
    elif speed_std > 1.0 and (speed_data[-1] - speed_data[0] > 2.0):
        classification = "Progressive Ramp-Up / Pyramid"
        visual_cue = "A clear diagonal gradient is visible in the speed channel."
    else:
        classification = "Variable or Recovery Session"
        visual_cue = "Low overall intensity and muted colors in the visual fingerprint."

    neighbors_text = ""
    if nearest_neighbors:
        for i, neighbor in enumerate(nearest_neighbors):
            neighbor_id_suffix = neighbor['_id'].split('_')[-1]
            neighbors_text += f"- Neighbor {i+1}: Workout #{neighbor_id_suffix} (Score: {neighbor['score']:.4f})\n"
    else:
        neighbors_text = "- No close 'Workout Twins' found in the database."

    prompt = f"""Analyze this structured workout data and provide a concise, qualitative summary (maximum 3 sentences) from the perspective of a **Workout Radiologist**. Focus on the *pattern* and *function* of the effort.

**Workout ID:** {doc['_id']}
**Primary Classification:** {classification}
**Quantitative Metrics:**
- Duration: {doc['duration_minutes']} minutes
- Avg HR / Max HR: {hr_avg:.1f} bpm / {hr_max:.1f} bpm
- Total Calories: {cal_total:.0f} kcal
- Speed Standard Deviation: {speed_std:.2f} kph

**Visual Pattern Insight:**
- {visual_cue}

**Vector Search Results (Workout Twins):**
{neighbors_text}
"""
    return classification, prompt.strip()


@app.get('/', response_class=HTMLResponse)
async def show_gallery():
    """
    Renders the main gallery page by reading templates/index.html,
    replacing placeholders, then returning the formatted HTML.
    """
    collection_images_html = []
    collection = db.workouts

    try:
        workouts_cursor = collection.find({}).sort("_id", ASCENDING)
        workouts = await workouts_cursor.to_list(length=200)
    except Exception as e:
        logger.error(f"Error fetching workouts for gallery: {e}")
        workouts = []
        collection_images_html.append(f"<p>Error loading workouts: {e}</p>")

    if not workouts and not collection_images_html:
        collection_images_html.append("<p>No workouts found. Click 'Generate'!</p>")

    for doc in workouts:
        try:
            workout_id_suffix_str = doc['_id'].split('_')[-1]
            workout_id_suffix = int(workout_id_suffix_str)
            arrays = generate_workout_viz_arrays(doc, 8)
            b64_img = encode_pil_image_to_base64(arrays['rgb_combined'], (128, 128), 'RGB')
            collection_images_html.append(f"""
            <div class="collection-item">
              <a href="/workout/{workout_id_suffix}">
                <img src="data:image/png;base64,{b64_img}" alt="Workout {workout_id_suffix}">
                <p>Workout #{workout_id_suffix}</p>
              </a>
            </div>
            """)
        except (ValueError, IndexError, KeyError, TypeError) as e:
            logger.warning(f"Skipping workout display due to error (ID: {doc.get('_id', 'N/A')}): {e}")
            continue

    # Read the external index.html template
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            template_str = f.read()
        # Replace placeholders
        template_str = template_str.replace("{{collection_images_html}}", "".join(collection_images_html))
        return HTMLResponse(content=template_str)
    except Exception as e:
        logger.error(f"Error reading or processing index.html template: {e}")
        raise HTTPException(status_code=500, detail="Error rendering index page.")


@app.get('/workout/{workout_id}', response_class=HTMLResponse)
async def show_workout_detail(workout_id: int):
    """
    Fetches a specific workout, runs Atlas Vector Search for neighbors,
    and displays detail by populating 'templates/detail.html'.
    """
    doc_id = f"workout_6b421a9c_{workout_id}"
    collection = db.workouts

    try:
        workout_doc = await collection.find_one({"_id": doc_id})
    except Exception as e:
        logger.error(f"DB error fetching workout {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error fetching workout.")

    if not workout_doc:
        raise HTTPException(status_code=404, detail=f"Workout '{doc_id}' not found.")

    ai_classification = workout_doc.get("ai_classification", PLACEHOLDER_CLASSIFICATION)
    ai_summary = workout_doc.get("ai_summary", PLACEHOLDER_SUMMARY)
    llm_analysis_prompt = workout_doc.get("llm_analysis_prompt", PLACEHOLDER_PROMPT)

    nearest_neighbors = []
    ai_neighbors_html = ""
    summary_is_pending = (ai_summary == PLACEHOLDER_SUMMARY or ai_classification == PLACEHOLDER_CLASSIFICATION)

    if "workout_vector" in workout_doc and isinstance(workout_doc["workout_vector"], list) and len(workout_doc["workout_vector"]) == 192:
        current_vector = workout_doc["workout_vector"]
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "workout_vector",
                    "queryVector": current_vector,
                    "numCandidates": 100,
                    "limit": 3,
                    "filter": {
                        "_id": {"$ne": doc_id}
                    }
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        try:
            neighbors_cursor = collection.aggregate(pipeline)
            nearest_neighbors = await neighbors_cursor.to_list(None)
            if nearest_neighbors:
                ai_neighbors_html_items = []
                for neighbor in nearest_neighbors:
                    neighbor_id = neighbor['_id']
                    neighbor_score = neighbor['score']
                    try:
                        neighbor_suffix = int(neighbor_id.split('_')[-1])
                        ai_neighbors_html_items.append(
                            f'<li><a href="/workout/{neighbor_suffix}">Workout #{neighbor_suffix}</a> (Similarity Score: {neighbor_score:.4f})</li>'
                        )
                    except (ValueError, IndexError):
                        ai_neighbors_html_items.append(
                            f'<li>Neighbor {neighbor_id} (Score: {neighbor_score:.4f})</li>'
                        )
                ai_neighbors_html = "".join(ai_neighbors_html_items)
            else:
                ai_neighbors_html = "<p>No other similar workouts found in this collection.</p>"

            if summary_is_pending:
                ai_classification, llm_analysis_prompt = analyze_time_series_features(workout_doc, nearest_neighbors)

        except OperationFailure as e:
            logger.error(f"OperationFailure during vector search for {doc_id}: {e.details}")
            ai_neighbors_html = f"<p><b>Database error during vector search.</b> Index '{VECTOR_INDEX_NAME}' might be building or failed. <br>Details: {e.details.get('errmsg', str(e))}</p>"
        except Exception as e:
            logger.error(f"Unexpected error during vector search for {doc_id}: {e}")
            ai_neighbors_html = f"<p><b>Application error during vector search:</b> {e}</p>"
    else:
        ai_neighbors_html = "<p>Vector data is missing or malformed for this workout.</p>"

    if summary_is_pending:
        # NOTE: The button's theme is now controlled by CSS in detail.html
        # We just provide the form and button structure.
        ai_analysis_button_html = f"""
        <form id="analyzeForm" action="/workout/{workout_id}/analyze" method="POST" style="margin: 0;">
          <button type="submit" id="analyzeBtn" class="control-btn" style="margin: 0; background-color: var(--accent-blue); color: white;">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                 fill="currentColor" viewBox="0 0 16 16" style="vertical-align: -2px; margin-right: 5px;">
              <path d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm-5.467 4.14C7.02 12.637 7.558 13
                   8 13c.448 0 .89-.37 1.341-.758.384-.33 1.164-.98 1.956-1.579.529-.396
                   .958-.87 1.253-1.412.308-.567.452-1.217.452-1.921 0-.663-.122-1.284-.367-1.841
                   -.247-.568-.62-1.11-1.12-1.583-.497-.47-1.127-.866-1.87-1.171C9.697 5.093 8.87
                   4.75 8 4.75c-.878 0-1.688.354-2.457.784-.735.41-1.353.94-1.854 1.572-.497.625
                   -.873 1.342-1.124 2.144-.25.808-.372 1.68-.372 2.616 0 .666.126 1.298.375 1.879
                   .248.568.618 1.107 1.117 1.582.497.47 1.127.865 1.87 1.171z"/>
              <path fill-rule="evenodd"
                    d="M8 15A7 7 0 1 0 8 1a7 7 0 0
                    0 0 14zM8 2A6 6 0 1 1
                    8 14 6 6 0 0 1 8 2z"/>
            </svg>
            Generate AI Summary
          </button>
        </form>
        """
    else:
        ai_analysis_button_html = '<span style="color: var(--atlas-green); font-weight: 600; font-size: 0.9em;">Analysis Complete</span>'

    viz_arrays = generate_workout_viz_arrays(workout_doc, 8)
    # Updated to generate dark-mode charts
    b64_chart_hr = generate_chart_base64(viz_arrays['raw_hr'], 'Heart Rate (64 mins)', '#ff3b30') # accent-red
    b64_chart_cal = generate_chart_base64(viz_arrays['raw_cal'], 'Calories / min', '#34c759') # green
    b64_chart_speed = generate_chart_base64(viz_arrays['raw_speed'], 'Speed (kph)', '#007aff') # accent-blue
    
    b64_combined = encode_pil_image_to_base64(viz_arrays['rgb_combined'], (256, 256), 'RGB')
    b64_r = encode_pil_image_to_base64(viz_arrays['channel_r_2d'], (128, 128), 'L', (255, 0, 0))
    b64_g = encode_pil_image_to_base64(viz_arrays['channel_g_2d'], (128, 128), 'L', (0, 255, 0))
    b64_b = encode_pil_image_to_base64(viz_arrays['channel_b_2d'], (128, 128), 'L', (0, 0, 255))

    doc_for_display = workout_doc.copy()
    if isinstance(doc_for_display.get("workout_vector"), list):
        vector_preview = str(doc_for_display["workout_vector"][:5])[1:-1]
        doc_for_display["workout_vector"] = f"[{vector_preview}, ... {len(workout_doc['workout_vector']) - 5} more elements]"

    doc_for_display.pop('experiment_id', None)
    doc_for_display.pop('ai_classification', None)
    doc_for_display.pop('ai_summary', None)
    doc_for_display.pop('llm_analysis_prompt', None)
    json_data_pretty = json.dumps(doc_for_display, indent=2, default=str)

    workout_type = workout_doc.get("workout_type", "N/A")
    session_tag = workout_doc.get("session_tag", "N/A")

    gear_used = workout_doc.get("gear_used", [])
    if gear_used:
        gear_used_html = "<p><b>Gear Used:</b></p><ul>"
        for g in gear_used:
            gear_used_html += f"<li>{json.dumps(g)}</li>"
        gear_used_html += "</ul>"
    else:
        gear_used_html = "<p><b>Gear Used:</b> <i>None</i></p>"

    sets_reps = workout_doc.get("sets_reps", [])
    if sets_reps:
        sets_reps_html = "<p><b>Sets/Reps:</b></p><ul>"
        for s in sets_reps:
            sets_reps_html += f"<li>{json.dumps(s)}</li>"
        sets_reps_html += "</ul>"
        if "rpe" in workout_doc:
            sets_reps_html += f"<p><b>RPE:</b> {workout_doc['rpe']}</p>"
    else:
        sets_reps_html = ""

    cycling_html = ""
    if workout_type == "Cycling":
        cadence_length = len(workout_doc.get("cadence_rpm", []))
        cycling_html = f"<p><b>Cadence RPM count:</b> {cadence_length} data points<br><b>Elevation Gain (m):</b> {workout_doc.get('elevation_gain_m',0)}</p>"

    yoga_html = ""
    if workout_type == "Yoga":
        focus_area = workout_doc.get("focus_area", "N/A")
        mood_rating = workout_doc.get("mood_rating", "N/A")
        yoga_html = f"<p><b>Focus Area:</b> {focus_area} <br><b>Mood Rating:</b> {mood_rating}</p>"

    post_notes = workout_doc.get("post_session_notes", {})
    post_session_notes_html = json.dumps(post_notes)

    # REFACTORED: Read from external template file
    try:
        with open("templates/detail.html", "r", encoding="utf-8") as f:
            template_str = f.read()
    except Exception as e:
        logger.error(f"Error reading or processing detail.html template: {e}")
        raise HTTPException(status_code=500, detail="Error rendering detail page.")

    # REFACTORED: Perform all string replacements on the template
    template_str = template_str.replace("{{workout_id}}", str(workout_id))
    template_str = template_str.replace("{{b64_combined}}", b64_combined)
    template_str = template_str.replace("{{norm_bounds_hr}}", f"{NORM_BOUNDS['heart_rate'][0]}-{NORM_BOUNDS['heart_rate'][1]}bpm")
    template_str = template_str.replace("{{norm_bounds_cal}}", f"{NORM_BOUNDS['calories_per_min'][0]}-{NORM_BOUNDS['calories_per_min'][1]}/min")
    template_str = template_str.replace("{{norm_bounds_speed}}", f"{NORM_BOUNDS['speed_kph'][0]}-{NORM_BOUNDS['speed_kph'][1]}kph")
    template_str = template_str.replace("{{json_data_pretty}}", json_data_pretty)
    template_str = template_str.replace("{{ai_classification}}", ai_classification)
    template_str = template_str.replace("{{ai_analysis_button_html}}", ai_analysis_button_html)
    template_str = template_str.replace("{{ai_summary}}", ai_summary)
    template_str = template_str.replace("{{vector_index_name}}", VECTOR_INDEX_NAME)
    template_str = template_str.replace("{{ai_neighbors_html}}", ai_neighbors_html)
    template_str = template_str.replace("{{b64_chart_hr}}", b64_chart_hr)
    template_str = template_str.replace("{{b64_r}}", b64_r)
    template_str = template_str.replace("{{b64_chart_cal}}", b64_chart_cal)
    template_str = template_str.replace("{{b64_g}}", b64_g)
    template_str = template_str.replace("{{b64_chart_speed}}", b64_chart_speed)
    template_str = template_str.replace("{{b64_b}}", b64_b)
    template_str = template_str.replace("{{workout_type}}", workout_type)
    template_str = template_str.replace("{{session_tag}}", session_tag)
    template_str = template_str.replace("{{gear_used_html}}", gear_used_html)
    template_str = template_str.replace("{{sets_reps_html}}", sets_reps_html)
    template_str = template_str.replace("{{cycling_html}}", cycling_html)
    template_str = template_str.replace("{{yoga_html}}", yoga_html)
    template_str = template_str.replace("{{post_session_notes_html}}", post_session_notes_html)
    template_str = template_str.replace("{{llm_analysis_prompt}}", llm_analysis_prompt)

    return HTMLResponse(content=template_str)


@app.post('/workout/{workout_id}/analyze', response_class=RedirectResponse)
async def analyze_workout_and_save(workout_id: int):
    doc_id = f"workout_6b421a9c_{workout_id}"
    collection = db.workouts

    if not OPENAI_API_KEY:
        error_detail = "OPENAI_API_KEY environment variable is not set. Cannot perform AI workout analysis."
        logger.error(error_detail)
        raise HTTPException(status_code=503, detail=error_detail)

    try:
        workout_doc = await collection.find_one({"_id": doc_id})
        if not workout_doc or "workout_vector" not in workout_doc:
            raise HTTPException(status_code=404, detail=f"Workout '{doc_id}' not found or missing vector data.")

        current_vector = workout_doc["workout_vector"]
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "workout_vector",
                    "queryVector": current_vector,
                    "numCandidates": 100,
                    "limit": 3,
                    "filter": {"_id": {"$ne": doc_id}}
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        try:
            neighbors_cursor = collection.aggregate(pipeline)
            nearest_neighbors = await neighbors_cursor.to_list(None)
        except OperationFailure:
            nearest_neighbors = []
            logger.warning("Vector search failed during analysis, proceeding without neighbors context.")

        classification, llm_prompt = analyze_time_series_features(workout_doc, nearest_neighbors)
        ai_summary = await call_openai_api(llm_prompt)

        update_data = {
            "ai_classification": classification,
            "ai_summary": ai_summary,
            "llm_analysis_prompt": llm_prompt
        }
        await collection.update_one({"_id": doc_id}, {"$set": update_data})
        logger.info(f"Successfully analyzed and updated workout: {doc_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during manual AI analysis for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Server error during analysis: {e.__class__.__name__}")

    return RedirectResponse(url=f'/workout/{workout_id}', status_code=303)


@app.post('/generate')
async def generate_new_workout():
    """
    Creates one new workout, calculates its vector, and inserts the document
    with AI analysis placeholders (LLM is called later).
    """
    async with generation_lock:
        try:
            collection = db.workouts
            latest_workout_cursor = collection.find({}, {"_id": 1}).sort("_id", DESCENDING).limit(1)
            latest_workout = await latest_workout_cursor.to_list(length=1)
            max_id_suffix = int(latest_workout[0]['_id'].split('_')[-1]) if latest_workout else -1
            new_id_suffix = max_id_suffix + 1

            new_doc = create_synthetic_apple_watch_data(new_id_suffix)
            new_doc["workout_vector"] = get_feature_vector(new_doc).tolist()

            new_doc["ai_classification"] = PLACEHOLDER_CLASSIFICATION
            new_doc["ai_summary"] = PLACEHOLDER_SUMMARY
            new_doc["llm_analysis_prompt"] = PLACEHOLDER_PROMPT

            await collection.insert_one(new_doc)
            logger.info(f"Generated and inserted new workout with placeholders: {new_doc['_id']}")
        except DuplicateKeyError:
            logger.error(f"DuplicateKeyError on insert: workout_6b421a9c_{new_id_suffix}")
        except Exception as e:
            logger.error(f"Error during workout generation: {e}")
            raise HTTPException(status_code=500, detail="Error during workout generation.")

    return RedirectResponse(url='/', status_code=303)


@app.post('/clear')
async def clear_collection():
    """Deletes all workouts in the collection."""
    collection = db.workouts
    try:
        delete_result = await collection.delete_many({})
        logger.info(f"Deleted {delete_result.deleted_count} workouts from the collection.")
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
    return RedirectResponse(url='/', status_code=303)


async def seed_database(collection: CollectionProxy, num_to_seed: int = 20):
    logger.info(f"Seeding collection with {num_to_seed} workouts...")
    docs_to_insert = []
    for i in range(num_to_seed):
        try:
            doc = create_synthetic_apple_watch_data(i)
            doc["workout_vector"] = get_feature_vector(doc).tolist()
            doc["ai_classification"] = PLACEHOLDER_CLASSIFICATION
            doc["ai_summary"] = PLACEHOLDER_SUMMARY
            doc["llm_analysis_prompt"] = PLACEHOLDER_PROMPT
            docs_to_insert.append(doc)
        except Exception as e:
            logger.error(f"Error creating synthetic data for workout index {i}: {e}")

    if docs_to_insert:
        try:
            insert_result = await collection.insert_many(docs_to_insert, ordered=False)
            logger.info(f"Attempted to insert {len(docs_to_insert)} workouts. Acknowledged inserts: {len(insert_result.inserted_ids)}.")
        except Exception as e:
            logger.error(f"Error bulk inserting seeded documents: {e}")
    else:
        logger.warning("No documents were generated for seeding.")


@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI app starting up...")
    num_seed_entries = 20
    collection = db.workouts
    index_manager = collection.index_manager
    index_ready = False
    needs_seeding = False

    MAX_STARTUP_RETRIES = 5
    RETRY_DELAY_SECONDS = 10

    try:
        # --- Cleaned up startup checks ---
        # Create templates directory if it doesn't exist
        if not os.path.exists("templates"):
            os.makedirs("templates")
            logger.info("Created 'templates' directory.")
        
        # Check for template files and log if missing
        if not os.path.exists("templates/index.html"):
             logger.warning("File 'templates/index.html' is missing. The root URL '/' will fail.")
        if not os.path.exists("templates/detail.html"):
             logger.warning("File 'templates/detail.html' is missing. Workout detail pages will fail.")
        # --- End of cleaned up checks ---

        logger.info("Checking total document count...")
        count = await collection.count_documents({}, limit=1)
        needs_seeding = (count == 0)
        logger.info(f"Collection is {'empty (needs seeding)' if needs_seeding else 'not empty (skipping seed)'}.")

        for attempt in range(MAX_STARTUP_RETRIES):
            try:
                logger.info(
                    f"Ensuring Atlas Vector Search index '{VECTOR_INDEX_NAME}' exists (Attempt {attempt + 1}/{MAX_STARTUP_RETRIES})..."
                )
                index_ready = await index_manager.create_search_index(
                    name=VECTOR_INDEX_NAME,
                    definition=VECTOR_INDEX_DEF,
                    index_type="vectorSearch",
                    wait_for_ready=True,
                    timeout=AsyncAtlasIndexManager.DEFAULT_SEARCH_TIMEOUT
                )
                if index_ready:
                    logger.info(f"Vector index '{VECTOR_INDEX_NAME}' is ready.")
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1}: create_search_index returned False.")
                    if attempt == MAX_STARTUP_RETRIES - 1:
                        logger.critical(
                            f"Vector index '{VECTOR_INDEX_NAME}' did NOT become ready "
                            f"after {MAX_STARTUP_RETRIES} attempts."
                        )
                        break
            except (OperationFailure, TimeoutError, AutoReconnect) as e:
                logger.warning(f"Attempt {attempt + 1} failed: Error ensuring search index '{VECTOR_INDEX_NAME}': {e}")
                if attempt < MAX_STARTUP_RETRIES - 1:
                    logger.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                else:
                    logger.critical(
                        f"CRITICAL STARTUP ERROR: Failed to ensure search index '{VECTOR_INDEX_NAME}' "
                        f"after {MAX_STARTUP_RETRIES} attempts. Last error: {e}"
                    )
                    index_ready = False
                    break
            except Exception as e:
                logger.critical(f"CRITICAL UNEXPECTED STARTUP ERROR (Attempt {attempt + 1}): {e}", exc_info=True)
                index_ready = False
                break

        try:
            await index_manager.create_index("_id")
        except Exception as e:
            logger.error(f"Failed to ensure standard _id index: {e}")

        if needs_seeding:
            await seed_database(collection, num_to_seed=num_seed_entries)
        else:
            logger.info("Skipping database seeding as collection is not empty.")

    except Exception as e:
        logger.critical(f"CRITICAL UNEXPECTED STARTUP ERROR: {e}", exc_info=True)

    logger.info(f"Startup sequence complete. Vector Index Ready: {index_ready}. Application ready.")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    logger.info(f"Starting Uvicorn server on {host}:{port} with reload enabled...")
    logger.info(f"Access the application at http://localhost:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)