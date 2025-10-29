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
import random  # For polymorphic data generation
from datetime import datetime, timezone  # For 'start_time' field

from typing import (
    Optional, List, Mapping, Any, Dict, Union, Tuple, ClassVar
)

# --- FastAPI & MongoDB (Async) ---
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
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

# Use 'Agg' backend for Matplotlib in a headless environment (no GUI)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load environment variables (MONGO_URI, OPENAI_API_KEY) from .env file
load_dotenv()

# ############################################################################
# ############################################################################
# Part 2: Asynchronous MongoDB Proxy Wrapper (Non-Scoped)
# (Includes AsyncAtlasIndexManager and Proxy classes)
# ############################################################################
# ############################################################################

# --- ROBUST IMPORT FIX ---
# Handle potential import differences for GEO2DSPHERE
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
    
    This class provides a robust, high-level API for index operations,
    including 'wait_for_ready' polling logic to handle the asynchronous
    nature of Atlas index builds.
    """
    # Use __slots__ for minor performance gain (faster attribute access)
    __slots__ = ('_collection',)

    # --- Class-level constants for polling and timeouts ---
    DEFAULT_POLL_INTERVAL: ClassVar[int] = 5  # seconds
    DEFAULT_SEARCH_TIMEOUT: ClassVar[int] = 600  # 10 minutes
    DEFAULT_DROP_TIMEOUT: ClassVar[int] = 300   # 5 minutes

    def __init__(self, real_collection: AsyncIOMotorCollection):
        """
        Initializes the manager with a direct reference to a
        motor.motor_asyncio.AsyncIOMotorCollection.
        """
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
        Creates or updates an Atlas Search index.
        
        This method is idempotent. It checks if an index with the same name
        and definition already exists and is queryable. If it exists but the
        definition has changed, it triggers an update. If it's building,
        it waits. If it doesn't exist, it creates it.
        """
        # --- Pre-check: Ensure collection exists ---
        # Atlas Search Index creation can sometimes fail if the collection
        # doesn't exist yet, especially in a race condition.
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

                # --- Definition Change Check ---
                # Compare the provided definition with the 'latestDefinition'
                # from the existing index.
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
                # --- End Check ---

                if definition_changed:
                    # Definitions differ, trigger an update
                    logger.warning(f"Search index '{name}' definition has changed ({change_reason}). Triggering update...")
                    await self.update_search_index(
                        name=name,
                        definition=definition,
                        wait_for_ready=False # Wait logic handled below
                    )
                elif existing_index.get("queryable"):
                    # Index exists, is up-to-date, and ready
                    logger.info(f"Search index '{name}' is already queryable and definition is up-to-date.")
                    return True
                elif existing_index.get("status") == "FAILED":
                    # Index exists but is in a failed state
                    logger.error(
                        f"Search index '{name}' exists but is in a FAILED state. "
                        f"Manual intervention in Atlas UI may be required."
                    )
                    return False
                else:
                    # Index exists, is up-to-date, but not queryable (e.g., "PENDING", "STALE")
                    logger.info(
                        f"Search index '{name}' exists and is up-to-date, "
                        f"but not queryable (Status: {existing_index.get('status')}). Waiting..."
                    )
            
            else:
                # --- Create New Index ---
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
                    # Handle race condition where another process created the index
                    if "IndexAlreadyExists" in str(e) or "DuplicateIndexName" in str(e):
                        logger.warning(f"Race condition: Index '{name}' was created by another process.")
                    else:
                        logger.error(f"OperationFailure during search index creation for '{name}': {e.details}")
                        raise e

            # --- Wait for Ready ---
            # If requested, poll the index status until it's queryable
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
        """
        Retrieves the definition and status of a single search index by name
        using the $listSearchIndexes aggregation stage.
        """
        try:
            pipeline = [{"$listSearchIndexes": {"name": name}}]
            async for index_info in self._collection.aggregate(pipeline):
                # We expect only one or zero results
                return index_info
            return None
        except OperationFailure as e:
            logger.error(f"OperationFailure retrieving search index '{name}': {e.details}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving search index '{name}': {e}")
            return None

    async def list_search_indexes(self) -> List[Dict[str, Any]]:
        """Lists all Atlas Search indexes for the collection."""
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
        """
        Drops an Atlas Search index by name.
        """
        try:
            # Check if index exists before trying to drop
            if not await self.get_search_index(name):
                logger.info(f"Search index '{name}' does not exist. Nothing to drop.")
                return True

            await self._collection.drop_search_index(name=name)
            logger.info(f"Submitted request to drop search index '{name}'.")

            if wait_for_drop:
                return await self._wait_for_search_index_drop(name, timeout)
            return True
        except OperationFailure as e:
            # Handle race condition where index was already dropped
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
        """
        Updates the definition of an existing Atlas Search index.
        This will trigger a rebuild of the index.
        """
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
        """
        Private helper to poll the index status until it becomes
        queryable or fails.
        """
        start_time = time.time()
        logger.info(f"Waiting up to {timeout}s for search index '{name}' to become queryable...")

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.error(f"Timeout: Index '{name}' did not become queryable within {timeout}s.")
                raise TimeoutError(f"Index '{name}' did not become queryable within {timeout}s.")

            index_info = None
            try:
                # Poll for the index status
                index_info = await self.get_search_index(name)
            except (OperationFailure, AutoReconnect) as e:
                # Handle transient network/DB errors during polling
                logger.warning(f"DB Error during polling for index '{name}': {getattr(e, 'details', e)}. Retrying...")
            except Exception as e:
                logger.error(f"Unexpected error during polling for index '{name}': {e}. Retrying...")

            if index_info:
                status = index_info.get("status")
                if status == "FAILED":
                    # The build failed permanently
                    logger.error(f"Search index '{name}' failed to build (Status: FAILED). Check Atlas UI for details.")
                    raise Exception(f"Index build failed for '{name}'.")

                queryable = index_info.get("queryable")
                if queryable:
                    # Success!
                    logger.info(f"Search index '{name}' is queryable (Status: {status}).")
                    return True

                # Not ready yet, log and wait
                logger.info(f"Polling for '{name}'. Status: {status}. Queryable: {queryable}. Elapsed: {elapsed:.0f}s")
            else:
                # Index not found yet (can happen right after creation command)
                logger.info(f"Polling for '{name}'. Index not found yet (normal during creation). Elapsed: {elapsed:.0f}s")

            await asyncio.sleep(self.DEFAULT_POLL_INTERVAL)

    async def _wait_for_search_index_drop(self, name: str, timeout: int) -> bool:
        """
        Private helper to poll until an index is successfully dropped.
        """
        start_time = time.time()
        logger.info(f"Waiting up to {timeout}s for search index '{name}' to be dropped...")
        while True:
            if time.time() - start_time > timeout:
                logger.error(f"Timeout: Index '{name}' was not dropped within {timeout}s.")
                raise TimeoutError(f"Index '{name}' was not dropped within {timeout}s.")

            index_info = await self.get_search_index(name)
            if not index_info:
                # Success! Index is gone.
                logger.info(f"Search index '{name}' has been successfully dropped.")
                return True

            logger.debug(f"Polling for '{name}' drop. Still present. Elapsed: {time.time() - start_time:.0f}s")
            await asyncio.sleep(self.DEFAULT_POLL_INTERVAL)

    # --- Regular Database Index Methods ---
    # These methods wrap the standard Motor index commands for a
    # consistent async API with the search index methods.
    
    async def create_index(
        self,
        keys: Union[str, List[Tuple[str, Union[int, str]]]],
        **kwargs: Any
    ) -> str:
        """
        Creates a standard (non-search) database index.
        Idempotent: checks if the index already exists first.
        """
        if isinstance(keys, str):
            keys = [(keys, ASCENDING)]

        # Attempt to auto-generate the index name if not provided
        index_name = kwargs.get("name")
        if not index_name:
            try:
                # Use pymongo helper to generate the name PyMongo would use
                from pymongo.helpers import _index_list
                index_doc = MongoClient()._database._CommandBuilder._gen_index_doc(keys, kwargs)
                index_name = _index_list(index_doc['key'].items())
            except Exception:
                # Fallback name generation
                index_name = f"index_{'_'.join([k[0] for k in keys])}"
                logger.warning(f"Could not auto-generate index name, using fallback: {index_name}")

        try:
            # Check if index already exists
            existing_indexes = await self.list_indexes()
            for index in existing_indexes:
                if index.get("name") == index_name:
                    logger.info(f"Regular index '{index_name}' already exists.")
                    return index_name

            # Create the index
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
        """Helper to create a standard text index."""
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
        """Helper to create a standard 2dsphere index."""
        keys = [(field, GEO2DSPHERE)]
        if name:
            kwargs["name"] = name
        return await self.create_index(keys, **kwargs)

    async def drop_index(self, name: str):
        """Drops a standard (non-search) database index by name."""
        try:
            await self._collection.drop_index(name)
            logger.info(f"Successfully dropped regular index '{name}'.")
        except OperationFailure as e:
            # Handle case where index is already gone
            if "index not found" in str(e).lower():
                logger.info(f"Regular index '{name}' does not exist. Nothing to drop.")
            else:
                logger.error(f"Failed to drop regular index '{name}': {e.details}")
                raise
        except Exception as e:
            logger.error(f"Failed to drop regular index '{name}': {e}")
            raise

    async def list_indexes(self) -> List[Dict[str, Any]]:
        """Lists all standard (non-search) indexes on the collection."""
        try:
            return await self._collection.list_indexes().to_list(None)
        except Exception as e:
            logger.error(f"Error listing regular indexes: {e}")
            return []

    async def get_index(self, name: str) -> Optional[Dict[str, Any]]:
        """Gets a single standard index by name."""
        indexes = await self.list_indexes()
        return next((index for index in indexes if index.get("name") == name), None)


class CollectionProxy:
    """
    Wraps an AsyncIOMotorCollection to add the .index_manager property.
    
    This class proxies all attribute access to the real Motor collection
    (e.g., .find_one(), .aggregate()) while seamlessly attaching the
    AsyncAtlasIndexManager.
    """
    __slots__ = ('_collection', '_index_manager')

    def __init__(self, real_collection: AsyncIOMotorCollection):
        self._collection = real_collection
        self._index_manager: Optional[AsyncAtlasIndexManager] = None

    @property
    def index_manager(self) -> AsyncAtlasIndexManager:
        """
        Lazy-loads the AsyncAtlasIndexManager instance.
        """
        if self._index_manager is None:
            self._index_manager = AsyncAtlasIndexManager(self._collection)
        return self._index_manager

    def __getattr__(self, name: str) -> Any:
        """
        Proxies attribute access to the underlying Motor collection.
        This is what makes 'db.workouts.find_one()' work.
        """
        if name.startswith('_'):
            # Prevent proxying internal attributes
            return object.__getattribute__(self, name)
        # Delegate all other access to the real collection object
        return getattr(self._collection, name)


class DbProxy:
    """
    Wraps an AsyncIOMotorDatabase to provide non-scoped collection access.
    
    When an attribute (collection) is accessed (e.g., 'db.workouts'),
    it returns a 'CollectionProxy' wrapper instead of the raw collection.
    """
    __slots__ = ('_db', '_wrapper_cache')

    def __init__(self, real_db: AsyncIOMotorDatabase):
        self._db = real_db
        # Cache wrappers to avoid creating new ones on every access
        self._wrapper_cache: Dict[str, CollectionProxy] = {}

    def __getattr__(self, name: str) -> Union[CollectionProxy, Any]:
        """
        Proxies attribute access to the underlying Motor database.
        
        If the attribute is a collection, it wraps it in CollectionProxy.
        Otherwise, it returns the attribute directly (e.g., db.name).
        """
        if name in self._wrapper_cache:
            return self._wrapper_cache[name]

        real_attr = getattr(self._db, name)
        if isinstance(real_attr, AsyncIOMotorCollection):
            # It's a collection; wrap it and cache it
            wrapper = CollectionProxy(real_collection=real_attr)
            self._wrapper_cache[name] = wrapper
            return wrapper
        else:
            # It's not a collection (e.g., db.name); return it directly
            return real_attr


# --- Database & API Configuration ---

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable not set.")

# Initialize the async client and wrap the database
async_client = AsyncIOMotorClient(MONGO_URI)
real_db = async_client["workout_db"]
db = DbProxy(real_db=real_db)  # Use the proxy for all DB operations

VECTOR_INDEX_NAME = "workout_vector_index"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-3.5-turbo"

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. LLM features will be disabled until set.")

# Placeholders for documents before AI analysis is run
PLACEHOLDER_SUMMARY = "Click 'Generate AI Summary' to analyze this workout using OpenAI and MongoDB Vector Search context."
PLACEHOLDER_CLASSIFICATION = "Pending Analysis"
PLACEHOLDER_PROMPT = "Prompt context is generated on-the-fly and displayed here when the workout detail page loads, even before the summary is generated."

# Atlas Vector Search index definition
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
            "path": "_id"  # Example of a filterable field
        }
    ]
}

# --- Part 4: Data Generation Logic (Augmented for Polymorphic Data) ---

def create_synthetic_apple_watch_data(workout_id_suffix: int = 0):
    """
    Creates a polymorphic workout document with random variations.
    
    - Always includes a 64-length time_series for HR, Calories, Speed.
    - Polymorphic: Randomly assigns a 'workout_type' (Run, Strength,
      Cycling, Yoga) and adds type-specific fields (e.g., 'sets_reps',
      'cadence_rpm', 'focus_area').
    - Adds other rich metadata like 'gear_used' and 'post_session_notes'.
    """
    np.random.seed(workout_id_suffix)  # Use suffix for deterministic randomness
    t = np.linspace(0, 2 * np.pi, 64) # 64 data points (minutes)
    
    # Base values influenced by ID to create variety
    hr_base = 110 + (workout_id_suffix % 7) * 5
    cal_base = 6 + (workout_id_suffix % 5) * 1
    speed_base = 4.0 + (workout_id_suffix % 6) * 0.5

    # Generate base patterns with some noise
    hr_pattern = hr_base + 60 * np.sin(t - np.pi / 2 + np.random.rand() * 0.5) + np.random.rand(64) * 10
    cal_pattern = cal_base + 4 * np.sin(t - np.pi / 2 + np.random.rand() * 0.5) + np.random.rand(64) * 2

    # Generate different speed patterns for variety
    if workout_id_suffix % 4 == 0:
        # Interval-style speed
        speed_pattern = speed_base * (
            np.sin(t * 4 + np.random.rand() * 0.5) > 0.5
        ) + np.random.rand(64) * 0.5
    elif workout_id_suffix % 4 == 1:
        # Ramp-up speed
        speed_pattern = 3.0 + t * (speed_base / (2 * np.pi)) + np.random.rand(64) * 0.3
    else:
        # Steady-state speed
        speed_pattern = np.full(64, speed_base * 0.8) + np.random.rand(64) * 0.5

    # Simulate warm-up and cool-down
    speed_pattern[:5] = 2.0 + np.random.rand(5) * 0.5 # Warm-up
    speed_pattern[-5:] = 1.0 + np.random.rand(5) * 0.5 # Cool-down
    hr_pattern[:5] -= 20
    hr_pattern[-5:] -= 10

    # Ensure data is within realistic bounds
    hr_pattern = np.maximum(hr_pattern, 50)
    cal_pattern = np.maximum(cal_pattern, 0)
    speed_pattern = np.maximum(speed_pattern, 0)

    # --- Base Document Structure ---
    doc = {
        "_id": f"workout_6b421a9c_{workout_id_suffix}",
        "user_id": f"user_{789 + workout_id_suffix % 10}",
        "start_time": datetime(2025, 10, 27, 10, (10 + workout_id_suffix % 40), 0, tzinfo=timezone.utc),
        "duration_minutes": 64,
        "workout_type": "Outdoor Run", # Default type
        "time_series": {
            "heart_rate": list(np.round(hr_pattern, 2)),
            "calories_per_min": list(np.round(cal_pattern, 2)),
            "speed_kph": list(np.round(speed_pattern, 2))
        }
    }

    # --- Polymorphic Data Logic ---
    # Based on the ID, add different fields for different workout types.
    # This simulates real-world, schema-flexible data.
    type_idx = workout_id_suffix % 4
    if type_idx == 0:
        doc["workout_type"] = "Strength Training"
        doc["sets_reps"] = [
            {"exercise": "squat", "reps": 10, "weight_kg": 60},
            {"exercise": "bench", "reps": 8, "weight_kg": 70}
        ]
        doc["rpe"] = int(np.random.randint(5, 10))
    elif type_idx == 1:
        # keep as default "Outdoor Run"
        pass
    elif type_idx == 2:
        doc["workout_type"] = "Cycling"
        doc["cadence_rpm"] = list(np.round(np.random.rand(64) * 80 + 70, 2))
        doc["elevation_gain_m"] = int(np.random.randint(50, 501))
    else:
        doc["workout_type"] = "Yoga"
        doc["focus_area"] = np.random.choice(["Hips & Mobility", "Full Body Flow", "Restorative", "Power Yoga"])
        doc["mood_rating"] = int(np.random.randint(1, 6))

    # --- Additional Rich Metadata (Common to all) ---
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
    """
    Clips and normalizes a NumPy array to a 0-255 uint8 scale.
    """
    # Clip data to defined bounds (e.g., HR 50-200)
    clipped_data = np.clip(data, min_val, max_val)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(clipped_data, dtype=np.uint8)
    # Normalize to 0.0 - 1.0
    normalized = (clipped_data - min_val) / range_val
    # Scale to 0 - 255 and cast to integer
    return (normalized * 255).astype(np.uint8)


# --- Normalization Bounds ---
# These fixed bounds are crucial. They ensure that (e.g.) a
# heart rate of 150bpm *always* maps to the same color value.
NORM_BOUNDS = {
    "heart_rate": (50, 200),
    "calories_per_min": (0, 20),
    "speed_kph": (0, 15)
}

def generate_workout_viz_arrays(doc: dict, image_dim: int = 8) -> dict:
    """
    Takes a workout document and generates the raw and normalized
    data arrays used for visualization and vector creation.
    
    Returns a dictionary containing:
    - raw_hr, raw_cal, raw_speed (1D 64-element arrays)
    - channel_r_2d, channel_g_2d, channel_b_2d (2D 8x8 arrays, 0-255)
    - rgb_combined (3D 8x8x3 array, 0-255)
    """
    required_length = image_dim * image_dim # 8x8 = 64
    
    # --- Default/Error State ---
    # Return zeroed-out arrays if data is missing or malformed
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

        # Validate that all time-series have the required 64 points
        if not (len(hr_data) == required_length and len(cal_data) == required_length and len(speed_data) == required_length):
            raise ValueError(f"Time series data must have {required_length} elements.")

        # --- Normalization (The "Encoding" Step) ---
        # Map raw values (e.g., 50-200bpm) to pixel values (0-255)
        r_norm = normalize_data(hr_data, *NORM_BOUNDS["heart_rate"])
        g_norm = normalize_data(cal_data, *NORM_BOUNDS["calories_per_min"])
        b_norm = normalize_data(speed_data, *NORM_BOUNDS["speed_kph"])

        # --- Reshaping (The "Folding" Step) ---
        # Fold the 1D 64-element arrays into 2D 8x8 grids
        r_2d = r_norm.reshape(image_dim, image_dim)
        g_2d = g_norm.reshape(image_dim, image_dim)
        b_2d = b_norm.reshape(image_dim, image_dim)

        return {
            "raw_hr": hr_data, "raw_cal": cal_data, "raw_speed": speed_data,
            "channel_r_2d": r_2d, "channel_g_2d": g_2d, "channel_b_2d": b_2d,
            # Stack the 3 8x8 channels into one 8x8x3 RGB array
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
    """
    Converts a NumPy array into a base64-encoded PNG string.
    
    - 'RGB' mode: Expects an (H, W, 3) array.
    - 'L' mode: Expects an (H, W) array (grayscale).
    - 'L' mode with 'color': Tints the grayscale array to the given RGB color.
    """
    # A tiny 1x1 black pixel placeholder for error cases
    error_placeholder = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42m"
                       "NkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
    try:
        if mode == 'L' and color and img_array.ndim == 2:
            # --- Tinting Logic for Grayscale Channels ---
            # Create an empty (H, W, 3) array
            colored_array = np.zeros((*img_array.shape, 3), dtype=np.uint8)
            # Apply the tint color
            for i in range(3):
                colored_array[..., i] = 255
                if color[i] == 0:
                    colored_array[..., i] = 255 - img_array
                elif color[i] < 255:
                    colored_array[..., i] = 255 - ((255 - color[i]) * img_array // 255)
            img = Image.fromarray(colored_array, 'RGB')
        elif img_array.ndim == 3 and img_array.shape[2] == 3 and mode == 'RGB':
            # Standard (H, W, 3) RGB array
            img = Image.fromarray(img_array, 'RGB')
        elif img_array.ndim == 2 and mode == 'L':
            # Standard (H, W) Grayscale array
            img = Image.fromarray(img_array, 'L')
        else:
            raise ValueError(f"Unsupported array shape/mode: {img_array.shape}, mode='{mode}'")

        if resize_dim:
            # Resize using NEAREST to preserve the pixelated look
            img = img.resize(resize_dim, Image.NEAREST)

        # Save to in-memory buffer and encode
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return error_placeholder


def generate_chart_base64(data: np.ndarray, title: str, color: str) -> str:
    """
    Generates a simple Matplotlib line chart and returns it as a
    base64-encoded PNG string, styled for a dark UI.
    """
    # --- Matplotlib styling for dark mode ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    fig.patch.set_facecolor('#132A38') # Atlas dark blue
    ax.set_facecolor('#132A38')
    
    ax.plot(data, color=color, linewidth=2)
    ax.set_title(title, fontsize=10, color='#F9FAFB')
    ax.set_xlim(0, len(data) - 1 if len(data) > 1 else 1)
    
    # Hide axes and ticks for a clean "sparkline" look
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
        # Crucial: Close the plot to prevent memory leaks
        plt.close(fig)
        plt.style.use('default') # Reset style to default
        
# --- END DATA/VIZ FUNCTIONS ---


def get_feature_vector(doc: dict) -> np.ndarray:
    """
    Main "embedding" function.
    Takes a workout doc, generates the 8x8x3 RGB array,
    and flattens it into a 1D (192-element) vector.
    
    8 * 8 * 3 = 192
    """
    arrays = generate_workout_viz_arrays(doc, 8)
    # .flatten() converts the (8, 8, 3) array to a (192,) array
    return arrays['rgb_combined'].flatten()


# --- FastAPI App Initialization ---
app = FastAPI()
generation_lock = asyncio.Lock() # Prevents race conditions during generation


# --- LLM Functions ---

# --- MODIFIED: New System Prompt ---
async def call_openai_api(prompt: str) -> str:
    """
    Calls the OpenAI Chat Completions API with the provided prompt.
    """
    # This new system prompt is more advanced. It instructs the AI
    # to act as a diagnostician, specifically focusing on synthesizing
    # discrepancies between quantitative and qualitative data.
    system_prompt = (
        "You are a professional Workout Radiologist. Your job is to synthesize all provided data "
        "into a concise, qualitative summary (max 3 sentences) for a fitness expert. "
        "Pay special attention to the **Radiology Task** section, where you must analyze any "
        "**discrepancies** between the quantitative metrics and the user's qualitative metadata. "
        "Explain the *function* of the effort and provide a holistic diagnosis."
    )
    # --- END MODIFIED ---

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt} # The large, structured prompt
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            logger.info(f"Calling OpenAI API with model {OPENAI_MODEL}...")
            response = await client.post(OPENAI_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status() # Raise an exception for 4xx/5xx errors
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

# --- MODIFIED: This function is now the core "magic" ---
def analyze_time_series_features(doc: dict, nearest_neighbors: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Analyzes a workout document to generate a classification and
    a rich, structured prompt for the LLM.
    
    This is the core "RAG" (Retrieval-Augmented Generation) logic.
    It combines:
    1.  Computed metrics (NumPy)
    2.  Rich metadata (from the document)
    3.  NEW: Date/Time context
    4.  Vector search results (the "retrieval" part)
    5.  NEW: An explicit task to reconcile discrepancies
    
    ...into a single, powerful prompt.
    """
    
    # --- 1. Compute Quantitative Metrics ---
    ts = doc.get('time_series', {})
    hr_data = np.array(ts.get('heart_rate', [0]))
    cal_data = np.array(ts.get('calories_per_min', [0]))
    speed_data = np.array(ts.get('speed_kph', [0]))

    hr_avg = np.mean(hr_data)
    hr_max = np.max(hr_data)
    speed_std = np.std(speed_data) # Standard deviation of speed
    cal_total = np.sum(cal_data)

    # --- 2. Generate Quantitative Classification (from Stats) ---
    quant_classification = ""
    visual_cue = ""
    
    if speed_std > 2.5 and hr_max > 180:
        quant_classification = "High-Intensity Interval Training"
        visual_cue = "Visual pattern suggests high-contrast, jagged peaks (Horizontal Stripes)."
    elif speed_std < 0.5 and hr_avg > 130:
        quant_classification = "Steady State Aerobic Run"
        visual_cue = "Visual pattern suggests low contrast and smooth, uniform color."
    elif speed_std > 1.0 and (len(speed_data) > 1 and (speed_data[-1] - speed_data[0] > 2.0)):
        quant_classification = "Progressive Ramp-Up / Pyramid"
        visual_cue = "Visual pattern suggests a clear diagonal gradient in the speed channel."
    else:
        quant_classification = "Variable or Recovery Session"
        visual_cue = "Visual pattern suggests low overall intensity and muted colors."

    # --- 3. Generate Qualitative Classification (from Metadata) ---
    session_tag = doc.get("session_tag", "N/A")
    user_notes = doc.get("post_session_notes", {}).get("notes", "N/A").lower()
    qual_classification = "N/A" # Default
    
    if session_tag == "Threshold" or "harder" in user_notes or session_tag == "Tempo Pace":
        qual_classification = "High-Intensity Threshold Session"
    elif session_tag == "Race Day":
        qual_classification = "Peak Effort (Race)"
    elif session_tag == "Recovery" or "fatigue" in user_notes:
        qual_classification = "Low-Intensity Recovery"

    # --- 4. Determine Final Classification ---
    # We trust the user's qualitative tag more, if it exists
    final_classification = qual_classification if qual_classification != "N/A" else quant_classification

    # --- 5. Format Nearest Neighbors (Cleaner Logic) ---
    neighbors_text_lines = []
    if nearest_neighbors:
        for i, neighbor in enumerate(nearest_neighbors):
            neighbor_id_suffix = neighbor['_id'].split('_')[-1]
            n_type = neighbor.get('workout_type', 'N/A')
            n_tag = neighbor.get('session_tag', 'N/A')
            n_class = neighbor.get('ai_classification')
            
            neighbors_text_lines.append(
                f"- Neighbor {i+1}: Workout #{neighbor_id_suffix} (Score: {neighbor['score']:.4f})"
            )
            neighbors_text_lines.append(f"    - Type: {n_type} (Tag: {n_tag})")
            
            # Only add the "Pattern" line if the neighbor has been analyzed
            if n_class and n_class != PLACEHOLDER_CLASSIFICATION:
                neighbors_text_lines.append(f"    - Pattern: {n_class}")
    else:
        neighbors_text_lines.append("- No close 'Workout Twins' found in the database.")
    neighbors_text = "\n".join(neighbors_text_lines)

    # --- 6. Format Additional Polymorphic Metadata ---
    metadata_lines = []
    metadata_lines.append(f"- Workout Type: {doc.get('workout_type', 'N/A')}")
    metadata_lines.append(f"- Session Tag: {session_tag}") # Already defined
        
    if 'sets_reps' in doc and doc['sets_reps']:
        sets_str = ", ".join([f"{s.get('exercise', 'N/A')} ({s.get('reps', 'N/A')} reps @ {s.get('weight_kg', 'N/A')}kg)" for s in doc['sets_reps']])
        metadata_lines.append(f"- Strength Sets: {sets_str}")
    if 'rpe' in doc:
        metadata_lines.append(f"- RPE (1-10): {doc['rpe']}")
        
    if 'focus_area' in doc:
        metadata_lines.append(f"- Yoga Focus: {doc['focus_area']}")
    if 'mood_rating' in doc:
        metadata_lines.append(f"- Yoga Mood (1-5): {doc['mood_rating']}")
        
    if 'cadence_rpm' in doc and doc['cadence_rpm']:
        avg_cadence = np.mean(doc['cadence_rpm'])
        metadata_lines.append(f"- Avg Cadence: {avg_cadence:.1f} RPM")
    if 'elevation_gain_m' in doc:
        metadata_lines.append(f"- Elevation Gain: {doc['elevation_gain_m']}m")

    if 'gear_used' in doc and doc['gear_used']:
        gear_str = ", ".join([g.get('item', 'N/A') for g in doc['gear_used']])
        metadata_lines.append(f"- Gear Used: {gear_str}")
        
    if 'post_session_notes' in doc:
        hydration = doc['post_session_notes'].get('hydration_ml', 'N/A')
        metadata_lines.append(f"- Hydration: {hydration}ml")
        metadata_lines.append(f"- User Notes: {user_notes}")

    metadata_text = "\n".join(metadata_lines)

    # --- 7. NEW: Get Date/Time Context ---
    start_time = doc.get("start_time")
    day_of_week = "N/A"
    time_of_day = "N/A"
    time_formatted = "N/A"
    
    if isinstance(start_time, datetime):
        day_of_week = start_time.strftime("%A")
        time_formatted = start_time.strftime('%I:%M %p')
        hour = start_time.hour
        if 5 <= hour < 12:
            time_of_day = "Morning"
        elif 12 <= hour < 17:
            time_of_day = "Afternoon"
        elif 17 <= hour < 21:
            time_of_day = "Evening"
        else:
            time_of_day = "Night"

    # --- 8. NEW: Define the Radiology Task ---
    task_text = (
        f"**Quantitative Analysis (from metrics):** The time-series pattern is classified as '{quant_classification}'.\n"
        f"**Qualitative Analysis (from metadata):** The user tagged this as '{session_tag}' and noted '{user_notes}'."
    )
    # This is the "magic" part: explicitly point out the contradiction.
    if qual_classification != "N/A" and quant_classification != qual_classification:
        task_text += "\n\n**-> Your primary task is to diagnose this discrepancy.** Is the user's tag correct despite the metrics? Is the equipment faulty? Or is the quantitative model wrong? Provide your expert synthesis."
    else:
        task_text += "\n\n**-> Your primary task is to synthesize these aligned findings** into a holistic summary."

    # --- 9. Build the Final Augmented Prompt ---
    # This new structure is clearer for the LLM.
    prompt = f"""**Workout ID:** {doc['_id']}
**Final Classification:** {final_classification}

**Section 1: Quantitative & Qualitative Data**
**Key Metrics:**
- Duration: {doc['duration_minutes']} minutes
- Avg HR / Max HR: {hr_avg:.1f} bpm / {hr_max:.1f} bpm
- Total Calories: {cal_total:.0f} kcal
- Speed Standard Deviation: {speed_std:.2f} kph (low std = steady, high std = intervals)

**Full Metadata:**
{metadata_text} 

**Section 2: Contextual Data**
**Workout Timing:**
- Day: {day_of_week}
- Time: {time_of_day} ({time_formatted})

**Vector Search Results (Workout Twins):**
{neighbors_text}
{visual_cue}

**Section 3: Radiology Task**
{task_text}
"""
    # Return the final classification to be stored in the DB
    return final_classification, prompt.strip()


# --- FastAPI Endpoints ---

@app.get('/', response_class=HTMLResponse)
async def show_gallery():
    """
    Renders the main gallery page (templates/index.html).
    
    Fetches all workouts, generates their 8x8 visual fingerprints,
    and injects the HTML for the gallery grid into the template.
    """
    collection_images_html = []
    collection = db.workouts

    try:
        # Fetch all workouts, sorted by ID
        workouts_cursor = collection.find({}).sort("_id", ASCENDING)
        workouts = await workouts_cursor.to_list(length=200) # Limit to 200
    except Exception as e:
        logger.error(f"Error fetching workouts for gallery: {e}")
        workouts = []
        collection_images_html.append(f"<p>Error loading workouts: {e}</p>")

    if not workouts and not collection_images_html:
        collection_images_html.append("<p>No workouts found. Click 'Generate'!</p>")

    # --- Gallery Item Generation ---
    for doc in workouts:
        try:
            workout_id_suffix_str = doc['_id'].split('_')[-1]
            workout_id_suffix = int(workout_id_suffix_str)
            # Generate the 8x8x3 array
            arrays = generate_workout_viz_arrays(doc, 8)
            # Encode it as a base64 string, resized to 128x128
            b64_img = encode_pil_image_to_base64(arrays['rgb_combined'], (128, 128), 'RGB')
            # Create the HTML snippet
            collection_images_html.append(f"""
            <div class="collection-item">
              <a href="/workout/{workout_id_suffix}">
                <img src="data:image/png;base64,{b64_img}" alt="Workout {workout_id_suffix}">
                <p>Workout #{workout_id_suffix}</p>
              </a>
            </div>
            """)
        except (ValueError, IndexError, KeyError, TypeError) as e:
            # Handle malformed data gracefully
            logger.warning(f"Skipping workout display due to error (ID: {doc.get('_id', 'N/A')}): {e}")
            continue

    # --- Template Injection ---
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            template_str = f.read()
        # Replace the placeholder with the generated HTML
        template_str = template_str.replace("{{collection_images_html}}", "".join(collection_images_html))
        return HTMLResponse(content=template_str)
    except Exception as e:
        logger.error(f"Error reading or processing index.html template: {e}")
        raise HTTPException(status_code=500, detail="Error rendering index page.")


@app.get('/workout/{workout_id}', response_class=HTMLResponse)
async def show_workout_detail(workout_id: int):
    """
    Renders the workout detail page (templates/detail.html).
    
    1.  Fetches the specific workout document.
    2.  Runs a `$vectorSearch` aggregation to find its "Workout Twins".
    3.  If not already analyzed, generates the RAG prompt.
    4.  Generates all visualizations (charts, fingerprints).
    5.  Injects all data into the 'templates/detail.html' template.
    """
    doc_id = f"workout_6b421a9c_{workout_id}"
    collection = db.workouts

    # --- 1. Fetch Document ---
    try:
        workout_doc = await collection.find_one({"_id": doc_id})
    except Exception as e:
        logger.error(f"DB error fetching workout {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error fetching workout.")

    if not workout_doc:
        raise HTTPException(status_code=404, detail=f"Workout '{doc_id}' not found.")

    # Get existing AI data or use placeholders
    ai_classification = workout_doc.get("ai_classification", PLACEHOLDER_CLASSIFICATION)
    ai_summary = workout_doc.get("ai_summary", PLACEHOLDER_SUMMARY)
    llm_analysis_prompt = workout_doc.get("llm_analysis_prompt", PLACEHOLDER_PROMPT)

    nearest_neighbors = []
    ai_neighbors_html = ""
    summary_is_pending = (ai_summary == PLACEHOLDER_SUMMARY or ai_classification == PLACEHOLDER_CLASSIFICATION)

    # --- 2. Run Vector Search (if vector exists) ---
    if "workout_vector" in workout_doc and isinstance(workout_doc["workout_vector"], list) and len(workout_doc["workout_vector"]) == 192:
        current_vector = workout_doc["workout_vector"]
        
        # This pipeline retrieves the key metadata from neighbors
        # needed for the prompt and UI.
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "workout_vector",
                    "queryVector": current_vector,
                    "numCandidates": 100, # Check 100 nearest neighbors
                    "limit": 3,           # Return the top 3
                    "filter": {
                        "_id": {"$ne": doc_id} # Exclude the doc itself
                    }
                }
            },
            {
                # Request the fields we need for the prompt and UI
                "$project": {
                    "_id": 1,
                    "score": {"$meta": "vectorSearchScore"},
                    "workout_type": 1,
                    "session_tag": 1,
                    "ai_classification": 1 # The neighbor's analysis
                }
            }
        ]
        
        try:
            neighbors_cursor = collection.aggregate(pipeline)
            nearest_neighbors = await neighbors_cursor.to_list(None)
            
            # --- Build Neighbor HTML for UI (Cleaner Logic) ---
            if nearest_neighbors:
                ai_neighbors_html_items = []
                for neighbor in nearest_neighbors:
                    neighbor_id = neighbor['_id']
                    neighbor_score = neighbor['score']
                    try:
                        neighbor_suffix = int(neighbor_id.split('_')[-1])
                        # Get neighbor context for display
                        n_type = neighbor.get('workout_type', 'N/A')
                        n_tag = neighbor.get('session_tag', 'N/A') # Added tag
                        n_class = neighbor.get('ai_classification')
                        
                        # Build the contextual span
                        context_span_items = [f"Type: {n_type}", f"Tag: {n_tag}"]
                        # Only add "Pattern" if it's analyzed
                        if n_class and n_class != PLACEHOLDER_CLASSIFICATION:
                            context_span_items.append(f"Pattern: {n_class}")
                        
                        context_span = " | ".join(context_span_items)
                        
                        # Create the final HTML list item
                        ai_neighbors_html_items.append(
                            f'<li><a href="/workout/{neighbor_suffix}">Workout #{neighbor_suffix}</a>'
                            f'<span>({context_span})</span>'
                            f'<br>Similarity Score: {neighbor_score:.4f}</li>'
                        )
                    except (ValueError, IndexError):
                        ai_neighbors_html_items.append(
                            f'<li>Neighbor {neighbor_id} (Score: {neighbor_score:.4f})</li>'
                        )
                ai_neighbors_html = "".join(ai_neighbors_html_items)
            else:
                ai_neighbors_html = "<p>No other similar workouts found in this collection.</p>"

            # --- 3. Generate RAG Prompt (if needed) ---
            if summary_is_pending:
                # This function now receives the *rich* nearest_neighbors
                # list and will generate the *smarter* prompt.
                # It returns the NEW classification and the NEW prompt.
                ai_classification, llm_analysis_prompt = analyze_time_series_features(workout_doc, nearest_neighbors)

        except OperationFailure as e:
            # Handle common DB errors (e.g., index is building)
            logger.error(f"OperationFailure during vector search for {doc_id}: {e.details}")
            ai_neighbors_html = f"<p><b>Database error during vector search.</b> Index '{VECTOR_INDEX_NAME}' might be building or failed. <br>Details: {e.details.get('errmsg', str(e))}</p>"
        except Exception as e:
            logger.error(f"Unexpected error during vector search for {doc_id}: {e}")
            ai_neighbors_html = f"<p><b>Application error during vector search:</b> {e}</p>"
    else:
        ai_neighbors_html = "<p>Vector data is missing or malformed for this workout.</p>"

    # --- AI Analysis Button HTML ---
    # Show "Generate" button if pending, otherwise "Analysis Complete"
    if summary_is_pending:
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

    # --- 4. Generate All Visualizations ---
    viz_arrays = generate_workout_viz_arrays(workout_doc, 8)
    b64_chart_hr = generate_chart_base64(viz_arrays['raw_hr'], 'Heart Rate (64 mins)', '#ff3b30') # accent-red
    b64_chart_cal = generate_chart_base64(viz_arrays['raw_cal'], 'Calories / min', '#34c759') # green
    b64_chart_speed = generate_chart_base64(viz_arrays['raw_speed'], 'Speed (kph)', '#007aff') # accent-blue
    
    # Generate main fingerprint (resized)
    b64_combined = encode_pil_image_to_base64(viz_arrays['rgb_combined'], (256, 256), 'RGB')
    # Generate tinted channels
    b64_r = encode_pil_image_to_base64(viz_arrays['channel_r_2d'], (128, 128), 'L', (255, 0, 0))
    b64_g = encode_pil_image_to_base64(viz_arrays['channel_g_2d'], (128, 128), 'L', (0, 255, 0))
    b64_b = encode_pil_image_to_base64(viz_arrays['channel_b_2d'], (128, 128), 'L', (0, 0, 255))

    # --- 5. Prepare Data for Template Injection ---
    
    # Create a copy of the doc for JSON display
    doc_for_display = workout_doc.copy()
    # Truncate the vector for readability in the UI
    if isinstance(doc_for_display.get("workout_vector"), list):
        vector_preview = str(doc_for_display["workout_vector"][:5])[1:-1]
        doc_for_display["workout_vector"] = f"[{vector_preview}, ... {len(workout_doc['workout_vector']) - 5} more elements]"
    # Remove fields that are displayed elsewhere
    doc_for_display.pop('experiment_id', None)
    doc_for_display.pop('ai_classification', None)
    doc_for_display.pop('ai_summary', None)
    doc_for_display.pop('llm_analysis_prompt', None)
    # Pretty-print the JSON
    json_data_pretty = json.dumps(doc_for_display, indent=2, default=str)

    # --- Polymorphic HTML Generation ---
    # Generate specific HTML snippets for different workout types
    workout_type = workout_doc.get("workout_type", "N/A")
    session_tag = workout_doc.get("session_tag", "N/A")

    gear_used = workout_doc.get("gear_used", [])
    gear_used_html = ""
    if gear_used:
        gear_used_html = "<p><b>Gear Used:</b></p><ul>"
        for g in gear_used:
            gear_used_html += f"<li>{json.dumps(g)}</li>"
        gear_used_html += "</ul>"
    
    sets_reps = workout_doc.get("sets_reps", [])
    sets_reps_html = ""
    if sets_reps:
        sets_reps_html = "<p><b>Sets/Reps:</b></p><ul>"
        for s in sets_reps:
            sets_reps_html += f"<li>{json.dumps(s)}</li>"
        sets_reps_html += "</ul>"
        if "rpe" in workout_doc:
            sets_reps_html += f"<p><b>RPE:</b> {workout_doc['rpe']}</p>"

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

    # --- Read and Populate Template ---
    try:
        with open("templates/detail.html", "r", encoding="utf-8") as f:
            template_str = f.read()
    except Exception as e:
        logger.error(f"Error reading or processing detail.html template: {e}")
        raise HTTPException(status_code=500, detail="Error rendering detail page.")

    # Perform all string replacements
    template_str = template_str.replace("{{workout_id}}", str(workout_id))
    template_str = template_str.replace("{{b64_combined}}", b64_combined)
    template_str = template_str.replace("{{norm_bounds_hr}}", f"{NORM_BOUNDS['heart_rate'][0]}-{NORM_BOUNDS['heart_rate'][1]}bpm")
    template_str = template_str.replace("{{norm_bounds_cal}}", f"{NORM_BOUNDS['calories_per_min'][0]}-{NORM_BOUNDS['calories_per_min'][1]}/min")
    template_str = template_str.replace("{{norm_bounds_speed}}", f"{NORM_BOUNDS['speed_kph'][0]}-{NORM_BOUNDS['speed_kph'][1]}kph")
    template_str = template_str.replace("{{json_data_pretty}}", json_data_pretty)
    # This now uses the *new* smart classification, even before saving
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
    # This now contains the new, structured prompt with the Radiology Task
    template_str = template_str.replace("{{llm_analysis_prompt}}", llm_analysis_prompt)

    return HTMLResponse(content=template_str)


@app.post('/workout/{workout_id}/analyze', response_class=RedirectResponse)
async def analyze_workout_and_save(workout_id: int):
    """
    POST endpoint triggered by the 'Generate AI Summary' button.
    
    This performs the same logic as the detail page (fetch doc,
    run vector search, generate prompt) but then *calls* the
    OpenAI API and saves the results back to the database.
    """
    doc_id = f"workout_6b421a9c_{workout_id}"
    collection = db.workouts

    # Check for API key before proceeding
    if not OPENAI_API_KEY:
        error_detail = "OPENAI_API_KEY environment variable is not set. Cannot perform AI workout analysis."
        logger.error(error_detail)
        raise HTTPException(status_code=503, detail=error_detail)

    try:
        # --- 1. Fetch Document ---
        workout_doc = await collection.find_one({"_id": doc_id})
        if not workout_doc or "workout_vector" not in workout_doc:
            raise HTTPException(status_code=404, detail=f"Workout '{doc_id}' not found or missing vector data.")

        # --- 2. Run Vector Search ---
        current_vector = workout_doc["workout_vector"]
        # The pipeline is identical to the one in show_workout_detail
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
                    "score": {"$meta": "vectorSearchScore"},
                    "workout_type": 1,
                    "session_tag": 1,
                    "ai_classification": 1
                }
            }
        ]
        try:
            neighbors_cursor = collection.aggregate(pipeline)
            nearest_neighbors = await neighbors_cursor.to_list(None)
        except OperationFailure:
            # If search fails (e.g., index not ready), proceed without neighbor context
            nearest_neighbors = []
            logger.warning("Vector search failed during analysis, proceeding without neighbors context.")

        # --- 3. Generate Prompt ---
        # This generates the smart, augmented prompt
        classification, llm_prompt = analyze_time_series_features(workout_doc, nearest_neighbors)
        
        # --- 4. Call OpenAI API ---
        # This uses the new system prompt and the new user prompt
        ai_summary = await call_openai_api(llm_prompt)

        # --- 5. Save Results to MongoDB ---
        update_data = {
            "ai_classification": classification, # Save the new smart classification
            "ai_summary": ai_summary,
            "llm_analysis_prompt": llm_prompt
        }
        await collection.update_one({"_id": doc_id}, {"$set": update_data})
        logger.info(f"Successfully analyzed and updated workout: {doc_id}")
        
    except HTTPException:
        # Re-raise HTTPExceptions (like 503 from API key check)
        raise
    except Exception as e:
        logger.error(f"Error during manual AI analysis for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Server error during analysis: {e.__class__.__name__}")

    # Redirect back to the detail page (which will now show the results)
    return RedirectResponse(url=f'/workout/{workout_id}', status_code=303)


@app.post('/generate')
async def generate_new_workout():
    """
    Generates a single new workout, calculates its vector,
    and inserts it into the database with placeholder AI fields.
    """
    async with generation_lock:
        try:
            collection = db.workouts
            # Find the highest existing workout ID suffix
            latest_workout_cursor = collection.find({}, {"_id": 1}).sort("_id", DESCENDING).limit(1)
            latest_workout = await latest_workout_cursor.to_list(length=1)
            max_id_suffix = int(latest_workout[0]['_id'].split('_')[-1]) if latest_workout else -1
            new_id_suffix = max_id_suffix + 1

            # Create the document
            new_doc = create_synthetic_apple_watch_data(new_id_suffix)
            # Generate the vector
            new_doc["workout_vector"] = get_feature_vector(new_doc).tolist()

            # Add placeholder fields
            new_doc["ai_classification"] = PLACEHOLDER_CLASSIFICATION
            new_doc["ai_summary"] = PLACEHOLDER_SUMMARY
            new_doc["llm_analysis_prompt"] = PLACEHOLDER_PROMPT

            await collection.insert_one(new_doc)
            logger.info(f"Generated and inserted new workout with placeholders: {new_doc['_id']}")
        except DuplicateKeyError:
            # Handle potential (though unlikely) race condition
            logger.error(f"DuplicateKeyError on insert: workout_6b421a9c_{new_id_suffix}")
        except Exception as e:
            logger.error(f"Error during workout generation: {e}")
            raise HTTPException(status_code=500, detail="Error during workout generation.")

    # Redirect back to the main gallery
    return RedirectResponse(url='/', status_code=303)


@app.post('/clear')
async def clear_collection():
    """
    Deletes all documents from the 'workouts' collection.
    """
    collection = db.workouts
    try:
        delete_result = await collection.delete_many({})
        logger.info(f"Deleted {delete_result.deleted_count} workouts from the collection.")
    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
    # Redirect back to the main gallery
    return RedirectResponse(url='/', status_code=303)


async def seed_database(collection: CollectionProxy, num_to_seed: int = 20):
    """
    Generates and inserts a batch of synthetic workouts.
    This is only called on startup if the collection is empty.
    """
    logger.info(f"Seeding collection with {num_to_seed} workouts...")
    docs_to_insert = []
    for i in range(num_to_seed):
        try:
            doc = create_synthetic_apple_watch_data(i)
            # Generate vector and add placeholders
            doc["workout_vector"] = get_feature_vector(doc).tolist()
            doc["ai_classification"] = PLACEHOLDER_CLASSIFICATION
            doc["ai_summary"] = PLACEHOLDER_SUMMARY
            doc["llm_analysis_prompt"] = PLACEHOLDER_PROMPT
            docs_to_insert.append(doc)
        except Exception as e:
            logger.error(f"Error creating synthetic data for workout index {i}: {e}")

    if docs_to_insert:
        try:
            # Use insert_many for efficient bulk insertion
            insert_result = await collection.insert_many(docs_to_insert, ordered=False)
            logger.info(f"Attempted to insert {len(docs_to_insert)} workouts. Acknowledged inserts: {len(insert_result.inserted_ids)}.")
        except Exception as e:
            logger.error(f"Error bulk inserting seeded documents: {e}")
    else:
        logger.warning("No documents were generated for seeding.")


@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event handler.
    
    1.  Checks for required 'templates' directory and files.
    2.  Checks if the database collection is empty.
    3.  Ensures the Atlas Vector Search index exists (critical).
    4.  Ensures standard indexes exist.
    5.  If the collection was empty, seeds it with 20 workouts.
    """
    logger.info("FastAPI app starting up...")
    num_seed_entries = 20
    collection = db.workouts
    index_manager = collection.index_manager # Access via the proxy
    index_ready = False
    needs_seeding = False

    MAX_STARTUP_RETRIES = 5
    RETRY_DELAY_SECONDS = 10

    try:
        # --- 1. Check for 'templates' directory and files ---
        if not os.path.exists("templates"):
            os.makedirs("templates")
            logger.info("Created 'templates' directory.")
        
        if not os.path.exists("templates/index.html"):
             logger.warning("File 'templates/index.html' is missing. The root URL '/' will fail.")
        if not os.path.exists("templates/detail.html"):
             logger.warning("File 'templates/detail.html' is missing. Workout detail pages will fail.")

        # --- 2. Check if seeding is needed ---
        logger.info("Checking total document count...")
        # Use limit(1) for a fast count check, even on large collections
        count = await collection.count_documents({}, limit=1)
        needs_seeding = (count == 0)
        logger.info(f"Collection is {'empty (needs seeding)' if needs_seeding else 'not empty (skipping seed)'}.")

        # --- 3. Ensure Atlas Vector Search Index (with retries) ---
        # This is the most critical startup step.
        for attempt in range(MAX_STARTUP_RETRIES):
            try:
                logger.info(
                    f"Ensuring Atlas Vector Search index '{VECTOR_INDEX_NAME}' exists (Attempt {attempt + 1}/{MAX_STARTUP_RETRIES})..."
                )
                # Use the robust index manager method
                index_ready = await index_manager.create_search_index(
                    name=VECTOR_INDEX_NAME,
                    definition=VECTOR_INDEX_DEF,
                    index_type="vectorSearch",
                    wait_for_ready=True, # Wait for it to be queryable
                    timeout=AsyncAtlasIndexManager.DEFAULT_SEARCH_TIMEOUT
                )
                if index_ready:
                    logger.info(f"Vector index '{VECTOR_INDEX_NAME}' is ready.")
                    break # Success, exit retry loop
                else:
                    # This case should ideally not be hit if wait_for_ready=True
                    logger.warning(f"Attempt {attempt + 1}: create_search_index returned False.")
                    if attempt == MAX_STARTUP_RETRIES - 1:
                        logger.critical(
                            f"Vector index '{VECTOR_INDEX_NAME}' did NOT become ready "
                            f"after {MAX_STARTUP_RETRIES} attempts."
                        )
                        break
            except (OperationFailure, TimeoutError, AutoReconnect) as e:
                # Handle known, retry-able errors
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
                # Handle unexpected errors
                logger.critical(f"CRITICAL UNEXPECTED STARTUP ERROR (Attempt {attempt + 1}): {e}", exc_info=True)
                index_ready = False
                break

        # --- 4. Ensure Standard Indexes ---
        try:
            # Ensure a basic index on _id (good practice)
            await index_manager.create_index("_id")
        except Exception as e:
            logger.error(f"Failed to ensure standard _id index: {e}")

        # --- 5. Seed Database (if needed) ---
        if needs_seeding:
            await seed_database(collection, num_to_seed=num_seed_entries)
        else:
            logger.info("Skipping database seeding as collection is not empty.")

    except Exception as e:
        logger.critical(f"CRITICAL UNEXPECTED STARTUP ERROR: {e}", exc_info=True)

    logger.info(f"Startup sequence complete. Vector Index Ready: {index_ready}. Application ready.")


# --- Main execution ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0" # Listen on all interfaces
    logger.info(f"Starting Uvicorn server on {host}:{port} with reload enabled...")
    logger.info(f"Access the application at http://localhost:{port}")
    # 'reload=True' is great for development, auto-restarts on file changes
    uvicorn.run("main:app", host=host, port=port, reload=True)