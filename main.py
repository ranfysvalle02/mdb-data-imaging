# ---
# main.py: FastAPI Workout Analyzer with Async Vector Search and OpenAI LLM
#
# This file integrates:
# 1. FastAPI application logic.
# 2. The complete 'async_mongo_wrapper' module (non-scoped).
# 3. Asynchronous refactoring using 'motor' and 'async/await'.
# 4. 'AsyncAtlasIndexManager' for automatic index creation on startup.
# 5. **OpenAI API integration (via httpx) for manual AI workout summarization.**
# 6. Generation lock to prevent duplicate key errors.
# 7. Refined Vector Index definition.
# ---

# ---
# Part 0: Imports
# ---
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
import httpx # For asynchronous HTTP requests


# ---
# Part 1: Initial Setup (Logging, Matplotlib, Env)
# ---

# Configure logging FIRST
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Set logger for this file
logger = logging.getLogger(__name__)

# Use 'Agg' backend for non-GUI environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load .env file (for MONGO_URI and OPENAI_API_KEY)
load_dotenv()


# ###########################################################################
# ############################################################################
# Part 2: Asynchronous MongoDB Proxy Wrapper (Non-Scoped)
# (Includes AsyncAtlasIndexManager and Proxy classes)
#
# ###########################################################################
# ###########################################################################

# --- ROBUST IMPORT FIX ---
try:
  from pymongo import GEO2DSPHERE
except ImportError:
  logger.warning("Could not import GEO2DSPHERE from pymongo. Defining manually.")
  GEO2DSPHERE = "2dsphere"
# --- END FIX ---


# ###########################################################################
# ASYNCHRONOUS ATLAS INDEX MANAGER (Complete)
# ###########################################################################

class AsyncAtlasIndexManager:
  """
  Manages MongoDB Atlas Search indexes (Vector & Lucene) and standard
  database indexes with an asynchronous (Motor-native) interface.
  """
  __slots__ = ('_collection',)
  # Class-level constants for polling
  DEFAULT_POLL_INTERVAL: ClassVar[int] = 5 # seconds
  DEFAULT_SEARCH_TIMEOUT: ClassVar[int] = 600 # 10 minutes
  DEFAULT_DROP_TIMEOUT: ClassVar[int] = 300 # 5 minutes

  def __init__(self, real_collection: AsyncIOMotorCollection):
    """Initializes the manager with the real AsyncIOMotorCollection."""
    if not isinstance(real_collection, AsyncIOMotorCollection):
      raise TypeError(
        f"Expected AsyncIOMotorCollection, got {type(real_collection)}"
      )
    self._collection = real_collection

  # --- Atlas Search Index Methods (Vector & Lucene) ---
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
    # --- 1. Ensure collection exists ---
    try:
      coll_name = self._collection.name
      # Use list_collection_names for a lighter-weight check
      all_collections = await self._collection.database.list_collection_names()
      if coll_name not in all_collections:
        # This is a safe operation; does nothing if it already exists
        await self._collection.database.create_collection(coll_name)
        logger.info(f"Created collection '{coll_name}' as it did not exist.")
    except Exception as e:
      logger.error(f"Failed to ensure collection '{self._collection.name}' exists: {e}")
      # This is a prerequisite, so we must raise
      raise Exception(f"Failed to create prerequisite collection '{self._collection.name}': {e}")

    try:
      # --- 2. Check for existing index ---
      existing_index = await self.get_search_index(name)

      if existing_index:
        logger.info(f"Search index '{name}' already exists.")
        latest_def = existing_index.get("latestDefinition", {})

        # --- 3. ROBUST CHANGE DETECTION (THE FIX) ---
        definition_changed = False
        change_reason = ""

        if "fields" in definition and index_type.lower() == "vectorsearch":
          # This is a vectorSearch definition
          existing_fields = latest_def.get("fields")
          if existing_fields != definition["fields"]:
            definition_changed = True
            change_reason = "vector 'fields' definition differs."
            logger.debug(f"Index '{name}' change. New fields: {definition['fields']}. Existing fields: {existing_fields}")
        elif "mappings" in definition and index_type.lower() == "search":
          # This is a Lucene search definition
          existing_mappings = latest_def.get("mappings")
          if existing_mappings != definition["mappings"]:
            definition_changed = True
            change_reason = "Lucene 'mappings' definition differs."
            logger.debug(f"Index '{name}' change. New mappings: {definition['mappings']}. Existing mappings: {existing_mappings}")

        else:
          # This can happen if the definition is malformed or if
          # index_type doesn't match the definition keys.
          logger.warning(
            f"Index definition '{name}' has keys that don't match "
            f"index_type '{index_type}'. Cannot reliably check for changes."
          )

        # --- 3a. Trigger update if changed ---
        if definition_changed:
          logger.warning(f"Search index '{name}' definition has changed ({change_reason}). Triggering update...")
          await self.update_search_index(
            name=name,
            definition=definition,
            wait_for_ready=False # Wait is handled below
          )

        # --- 3b. Handle existing index (no changes) ---
        elif existing_index.get("queryable"):
          logger.info(f"Search index '{name}' is already queryable and definition is up-to-date.")
          return True
        elif existing_index.get("status") == "FAILED":
          logger.error(f"Search index '{name}' exists but is in a FAILED state. Manual intervention in Atlas UI may be required.")
          return False
        else:
          logger.info(f"Search index '{name}' exists and is up-to-date, but not queryable (Status: {existing_index.get('status')}). Waiting...")

      # --- 4. Create new index (if it didn't exist) ---
      else:
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
          # Handle race condition where another process created it
          if "IndexAlreadyExists" in str(e) or "DuplicateIndexName" in str(e):
            logger.warning(f"Race condition: Index '{name}' was created by another process. Proceeding to wait.")
          else:
            logger.error(f"OperationFailure during search index *creation* for '{name}': {e.details}")
            raise e # Re-raise

      # --- 5. Wait for index to be ready ---
      if wait_for_ready:
        return await self._wait_for_search_index_ready(name, timeout)
      return True # Return True if just submitted (wait_for_ready=False)

    except OperationFailure as e:
      logger.error(f"OperationFailure during search index creation/check for '{name}': {e.details}")
      raise
    except Exception as e:
      logger.error(f"An unexpected error occurred regarding search index '{name}': {e}")
      raise

  async def get_search_index(self, name: str) -> Optional[Dict[str, Any]]:
    """Retrieves the status for a single Atlas Search index."""
    try:
      # $listSearchIndexes is an aggregation stage
      pipeline = [{"$listSearchIndexes": {"name": name}}]
      async for index_info in self._collection.aggregate(pipeline):
        return index_info # Return the first (and only) result
      return None # Not found
    except OperationFailure as e:
      logger.error(f"OperationFailure retrieving search index '{name}': {e.details}")
      return None # Treat as not found on error for robustness
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
    """Drops an Atlas Search index by name."""
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
    """Updates the definition of an existing Atlas Search index."""
    try:
      logger.info(f"Updating search index '{name}'...")
      await self._collection.update_search_index(name=name, definition=definition)
      logger.info(f"Search index '{name}' update submitted. Rebuild initiated.")

      if wait_for_ready:
        # Note: Update triggers a rebuild, so we wait for 'queryable'
        return await self._wait_for_search_index_ready(name, timeout)
      return True
    except OperationFailure as e:
      logger.error(f"Error updating search index '{name}': {e.details}")
      raise
    except Exception as e:
      logger.error(f"Error updating search index '{name}': {e}")
      raise

  async def _wait_for_search_index_ready(self, name: str, timeout: int) -> bool:
    """Async polling helper for an index to become queryable."""
    start_time = time.time()
    logger.info(f"Waiting up to {timeout}s for search index '{name}' to become queryable...")

    while True:
      elapsed = time.time() - start_time
      if elapsed > timeout:
        logger.error(f"Timeout: Index '{name}' did not become queryable within {timeout}s.")
        raise TimeoutError(f"Index '{name}' did not become queryable within {timeout}s.")

      index_info = None
      try:
        # Use the robust getter
        index_info = await self.get_search_index(name)
      except (OperationFailure, AutoReconnect) as e:
        logger.warning(f"DB Error during polling for index '{name}': {getattr(e, 'details', e)}. Retrying...")
      except Exception as e:
        logger.error(f"Unexpected error during polling for index '{name}': {e}. Retrying...")

      if index_info:
        status = index_info.get("status")
        # FAILED is a terminal state
        if status == "FAILED":
          logger.error(f"Search index '{name}' failed to build (Status: FAILED). Check Atlas UI for details.")
          raise Exception(f"Index build failed for '{name}'.")

        # queryable=True is the goal state
        queryable = index_info.get("queryable")
        if queryable:
          logger.info(f"Search index '{name}' is queryable (Status: {status}).")
          return True

        # Still building, log and wait
        logger.info(f"Polling for '{name}'. Status: {status}. Queryable: {queryable}. Elapsed: {elapsed:.0f}s")

      else:
        # This is normal right after a create request
        logger.info(f"Polling for '{name}'. Index not found yet (normal during creation). Elapsed: {elapsed:.0f}s")

      await asyncio.sleep(self.DEFAULT_POLL_INTERVAL)

  async def _wait_for_search_index_drop(self, name: str, timeout: int) -> bool:
    """Async polling helper for an index to be dropped."""
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
    """Creates a standard database index, skipping if it already exists."""
    if isinstance(keys, str):
      keys = [(keys, ASCENDING)]

    # We must manually generate the name to check for existence
    index_name = kwargs.get("name")
    if not index_name:
      # Use PyMongo's internal helper to generate the name
      try:
        from pymongo.helpers import _index_list
        index_doc = MongoClient()._database._CommandBuilder._gen_index_doc(keys, kwargs)
        index_name = _index_list(index_doc['key'].items())
      except Exception:
        # Fallback name if helper fails
        index_name = f"index_{'_'.join([k[0] for k in keys])}"
        logger.warning(f"Could not auto-generate index name, using fallback: {index_name}")

    try:
      # Check existence first to avoid errors and unnecessary ops
      existing_indexes = await self.list_indexes()
      for index in existing_indexes:
        if index.get("name") == index_name:
          # Check if definition is identical (optional but robust)
          # This is complex, for simplicity we just check name
          logger.info(f"Regular index '{index_name}' already exists.")
          return index_name

      # If we are here, the index does not exist by that name
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
    self, fields: List[str], weights: Optional[Dict[str, int]] = None, name: str = "text_index", **kwargs: Any
  ) -> str:
    """Helper to create a standard text index."""
    keys = [(field, TEXT) for field in fields]
    if weights: kwargs["weights"] = weights
    if name: kwargs["name"] = name
    return await self.create_index(keys, **kwargs)

  async def create_geo_index(
    self, field: str, name: Optional[str] = None, **kwargs: Any
  ) -> str:
    """Helper to create a 2dsphere index."""
    keys = [(field, GEO2DSPHERE)]
    if name: kwargs["name"] = name
    return await self.create_index(keys, **kwargs)

  async def drop_index(self, name: str):
    """Drops a standard database index by name."""
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
    """Lists all standard database indexes."""
    try:
      return await self._collection.list_indexes().to_list(None)
    except Exception as e:
      logger.error(f"Error listing regular indexes: {e}")
      return []

  async def get_index(self, name: str) -> Optional[Dict[str, Any]]:
    """Retrieves the specification for a standard index."""
    indexes = await self.list_indexes()
    return next((index for index in indexes if index.get("name") == name), None)


# ###########################################################################
# NON-SCOPED PROXY CLASSES (Complete)
# ###########################################################################

class CollectionProxy:
  """Wraps an AsyncIOMotorCollection for index management access and basic proxying."""
  __slots__ = ('_collection', '_index_manager')

  def __init__(self, real_collection: AsyncIOMotorCollection):
    self._collection = real_collection
    self._index_manager: Optional[AsyncAtlasIndexManager] = None

  @property
  def index_manager(self) -> AsyncAtlasIndexManager:
    """Gets the AsyncAtlasIndexManager."""
    if self._index_manager is None:
      self._index_manager = AsyncAtlasIndexManager(self._collection)
    return self._index_manager

  # Use __getattr__ to proxy all motor methods directly without modification
  def __getattr__(self, name: str) -> Any:
    """Proxies all attribute/method calls to the real collection."""
    if name.startswith('_'):
      # Allow internal access but prevent accidental public access to private fields of the proxy itself
      return object.__getattribute__(self, name)
    return getattr(self._collection, name)


class DbProxy:
  """Wraps an AsyncIOMotorDatabase to provide non-scoped collection access."""
  __slots__ = ('_db', '_wrapper_cache')

  def __init__(self, real_db: AsyncIOMotorDatabase):
    self._db = real_db
    self._wrapper_cache: Dict[str, CollectionProxy] = {}

  def __getattr__(self, name: str) -> Union[CollectionProxy, Any]:
    """Proxies attribute access, returning CollectionProxies for collections."""
    if name in self._wrapper_cache:
      return self._wrapper_cache[name]

    real_attr = getattr(self._db, name)

    if isinstance(real_attr, AsyncIOMotorCollection):
      wrapper = CollectionProxy(real_collection=real_attr)
      self._wrapper_cache[name] = wrapper
      return wrapper
    else:
      return real_attr


# ###########################################################################
# --- END OF PROXY CODE ---
# ###########################################################################

# ---
# Part 3: Async MongoDB Connection & LLM Configuration
# ---
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
  raise ValueError("MONGO_URI environment variable not set.")

async_client = AsyncIOMotorClient(MONGO_URI)
real_db = async_client["workout_db"]

# Create the global non-scoped database proxy
db = DbProxy(real_db=real_db)

# Define index constants
VECTOR_INDEX_NAME = "workout_vector_index"

# --- OPENAI CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-3.5-turbo"

if not OPENAI_API_KEY:
  logger.warning("OPENAI_API_KEY not set. LLM features will be disabled until set.")
# --- END OPENAI CONFIGURATION ---

# --- Placeholder constants for manual analysis ---
PLACEHOLDER_SUMMARY = "Click 'Generate AI Summary' to analyze this workout using OpenAI and MongoDB Vector Search context."
PLACEHOLDER_CLASSIFICATION = "Pending Analysis"
PLACEHOLDER_PROMPT = "Prompt context is generated on-the-fly and displayed here when the workout detail page loads, even before the summary is generated."
# --- END Placeholder constants ---

# ---  REFINED VECTOR INDEX DEFINITION (NON-SCOPED)  ---
VECTOR_INDEX_DEF = {
  "fields": [
    # --- Vector Field Definition ---
    {
      "type": "vector",
      "path": "workout_vector",     # The field containing the vector array
      "numDimensions": 192,       # Must match your vector size
      "similarity": "cosine"      # Or "cosine", "dotProduct"
    },
    # --- Fields for Pre-Filtering (only need _id for self-exclusion) ---
    {
      "type": "filter",
      "path": "_id"           # Field to use in the $vectorSearch 'filter' option
    }
  ]
}
# ---  END REFINED DEFINITION (NON-SCOPED)  ---


# ---
# Part 4: Data Generation Logic (Unchanged)
# ---
def create_synthetic_apple_watch_data(workout_id_suffix: int = 0):
  np.random.seed(workout_id_suffix) # Make generation deterministic per ID
  t = np.linspace(0, 2 * np.pi, 64)
  hr_base = 110 + (workout_id_suffix % 7) * 5
  cal_base = 6 + (workout_id_suffix % 5) * 1
  speed_base = 4.0 + (workout_id_suffix % 6) * 0.5

  hr_pattern = hr_base + 60 * np.sin(t - np.pi/2 + np.random.rand() * 0.5) + np.random.rand(64) * 10
  cal_pattern = cal_base + 4 * np.sin(t - np.pi/2 + np.random.rand() * 0.5) + np.random.rand(64) * 2

  # Vary speed pattern based on ID
  if workout_id_suffix % 4 == 0: # Interval-like
    speed_pattern = speed_base * (np.sin(t * 4 + np.random.rand() * 0.5) > 0.5) + np.random.rand(64) * 0.5
  elif workout_id_suffix % 4 == 1: # Ramp up
    speed_pattern = 3.0 + t * (speed_base / (2 * np.pi)) + np.random.rand(64) * 0.3
  else: # Steady state (mostly)
    speed_pattern = np.full(64, speed_base * 0.8) + np.random.rand(64) * 0.5

  # Simulate warm-up/cool-down
  speed_pattern[:5] = 2.0 + np.random.rand(5) * 0.5
  speed_pattern[-5:] = 1.0 + np.random.rand(5) * 0.5
  hr_pattern[:5] -= 20
  hr_pattern[-5:] -= 10

  # Ensure realistic bounds
  hr_pattern = np.maximum(hr_pattern, 50)
  cal_pattern = np.maximum(cal_pattern, 0)
  speed_pattern = np.maximum(speed_pattern, 0)

  doc = {
    "_id": f"workout_6b421a9c_{workout_id_suffix}",
    "user_id": f"user_{789 + workout_id_suffix % 10}",
    "start_time": f"2025-10-27T10:{10 + workout_id_suffix % 40:02d}:00Z",
    "duration_minutes": 64,
    "workout_type": "Outdoor Run",
    "time_series": {
      "heart_rate": list(np.round(hr_pattern, 2)),
      "calories_per_min": list(np.round(cal_pattern, 2)),
      "speed_kph": list(np.round(speed_pattern, 2))
    }
  }
  return doc


def normalize_data(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
  """Normalizes data to 0-255 based on expected bounds."""
  clipped_data = np.clip(data, min_val, max_val)
  range_val = max_val - min_val
  if range_val == 0:
    return np.zeros_like(clipped_data, dtype=np.uint8)
  normalized = (clipped_data - min_val) / range_val
  return (normalized * 255).astype(np.uint8)


# ---
# Part 5: Data-to-Image Processing (Unchanged)
# ---
NORM_BOUNDS = {
  "heart_rate": (50, 200),
  "calories_per_min": (0, 20),
  "speed_kph": (0, 15)
}


def generate_workout_viz_arrays(doc: dict, image_dim: int = 8) -> dict:
  """Processes workout doc into numpy arrays for visualization."""
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


# ---
# Part 6: Visualization & Encoding Helpers (Unchanged)
# ---
def encode_pil_image_to_base64(
  img_array: np.ndarray,
  resize_dim: Optional[Tuple[int, int]] = None,
  mode: str = 'RGB',
  color: Optional[Tuple[int, int, int]] = None # For coloring grayscale
) -> str:
  """Converts numpy array to Base64 PNG string via PIL."""
  error_placeholder = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
  try:
    # Handle grayscale coloring
    if mode == 'L' and color and img_array.ndim == 2:
      colored_array = np.zeros((*img_array.shape, 3), dtype=np.uint8)
      for i in range(3):
        colored_array[..., i] = 255 # Start with white background
        if color[i] == 0: # If target channel is 0 (black), invert pixel value
          colored_array[..., i] = 255 - img_array
        elif color[i] < 255: # If target channel is non-white, scale towards it
          colored_array[..., i] = 255 - ((255 - color[i]) * img_array // 255)
      img = Image.fromarray(colored_array, 'RGB')

    # Standard RGB or Grayscale
    elif img_array.ndim == 3 and img_array.shape[2] == 3 and mode == 'RGB':
      img = Image.fromarray(img_array, 'RGB')
    elif img_array.ndim == 2 and mode == 'L':
      img = Image.fromarray(img_array, 'L')
    else:
      raise ValueError(f"Unsupported array shape/mode: {img_array.shape}, mode='{mode}'")

    if resize_dim:
      img = img.resize(resize_dim, Image.NEAREST) # Use NEAREST for pixel art effect

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
  except Exception as e:
    logger.error(f"Error encoding image to base64: {e}")
    return error_placeholder


def generate_chart_base64(data: np.ndarray, title: str, color: str) -> str:
  """Generates a 1D line plot Base64 string."""
  fig, ax = plt.subplots(figsize=(4, 2), dpi=100) # Small chart size
  ax.plot(data, color=color, linewidth=2)
  ax.set_title(title, fontsize=10)
  ax.set_xlim(0, len(data) - 1 if len(data) > 1 else 1)
  ax.set_yticks([]) # Simplify y-axis
  plt.tight_layout()
  buf = io.BytesIO()
  try:
    fig.savefig(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')
  except Exception as e:
    logger.error(f"Error generating chart '{title}': {e}")
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
  finally:
    plt.close(fig) # Ensure figure is closed to free memory


# ---
# Part 7: "AI" (k-NN) Helper Function (Unchanged)
# ---
def get_feature_vector(doc: dict) -> np.ndarray:
  """Generates the 192-element (8x8x3) feature vector."""
  arrays = generate_workout_viz_arrays(doc, 8) # Force 8x8 dimension
  return arrays['rgb_combined'].flatten() # Flatten the 8x8x3 RGB array


# ---
# Part 8: The FastAPI App & HTML Templates (Updated)
# ---
app = FastAPI()

# Lock to prevent race conditions during generation
generation_lock = asyncio.Lock()

# --- HTML Templates (Updated for Manual Analysis) ---
PARTICLE_JS_SCRIPT = """<canvas id="particle-canvas" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;"></canvas><script>document.addEventListener("DOMContentLoaded", function() { const canvas = document.getElementById("particle-canvas"); const ctx = canvas.getContext("2d"); let particles = []; function resizeCanvas() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; } window.addEventListener("resize", resizeCanvas); resizeCanvas(); function createParticle() { const x = Math.random() * canvas.width; const y = Math.random() * canvas.height; const size = Math.random() * 2 + 1; const speedX = (Math.random() * 1 - 0.5) * 0.5; const speedY = (Math.random() * 1 - 0.5) * 0.5; const opacity = Math.random() * 0.5 + 0.2; particles.push({x, y, size, speedX, speedY, opacity}); } for (let i = 0; i < 50; i++) { createParticle(); } function animate() { ctx.clearRect(0, 0, canvas.width, canvas.height); particles.forEach((p, index) => { p.x += p.speedX; p.y += p.speedY; if (p.x < 0 || p.x > canvas.width || p.y < 0 || p.y > canvas.height) { particles.splice(index, 1); createParticle(); } ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2); ctx.fillStyle = `rgba(0, 122, 255, ${p.opacity})`; ctx.fill(); }); requestAnimationFrame(animate); } animate(); }); </script>"""

GALLERY_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Workout Collection Gallery</title><style>body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 0; background-color: #f9f9f9; }} .container {{ max-width: 1200px; margin: 20px auto; padding: 20px; background-color: rgba(255, 255, 255, 0.95); border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); position: relative; z-index: 1; }} h1 {{ color: #333; text-align: center; }} h2 {{ color: #555; text-align: center; font-weight: 400; margin-top: -10px; }} .controls {{ display: flex; justify-content: center; gap: 15px; margin: 20px 0; }} .control-btn {{ padding: 10px 20px; font-size: 1em; font-weight: 500; border-radius: 5px; text-decoration: none; cursor: pointer; border: none; transition: all 0.2s ease; }} .btn-generate {{ background-color: #007aff; color: #fff; }} .btn-generate:hover {{ background-color: #0056b3; }} .btn-clear {{ background-color: #e6e6e6; color: #d9534f; border: 1px solid #d4d4d4; }} .btn-clear:hover {{ background-color: #d9534f; color: #fff; border-color: #d9534f; }} .collection-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(128px, 1fr)); gap: 20px; margin-top: 30px; }} .collection-item {{ text-align: center; background-color: #fdfdfd; border-radius: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); font-size: 0.9em; color: #666; transition: all 0.2s ease-in-out; border: 1px solid #eee; }} .collection-item:hover {{ box-shadow: 0 4px 10px rgba(0,0,0,0.1); transform: translateY(-2px); }} .collection-item a {{ text-decoration: none; color: inherit; display: block; padding: 10px; }} .collection-item img {{ width: 128px; height: 128px; image-rendering: pixelated; margin-bottom: 8px; border: 1px solid #ddd; border-radius: 4px; }}</style></head><body><div class="container"><h1>Workout Collection Gallery</h1><h2>Workout Collection</h2><div class="controls"><form action="/generate" method="POST" style="margin: 0;"><button type="submit" class="control-btn btn-generate"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16" style="vertical-align: -2px; margin-right: 5px;"><path d="M8 4a.5.5 0 0 1 .5.5v3h3a.5.5 0 0 1 0 1h-3v3a.5.5 0 0 1-1 0v-3h-3a.5.5 0 0 1 0-1h3v-3A.5.5 0 0 1 8 4z"/></svg>Generate New Workout</button></form><form action="/clear" method="POST" style="margin: 0;" onsubmit="return confirm('Are you sure you want to delete ALL workouts?');"><button type="submit" class="control-btn btn-clear"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16" style="vertical-align: -2px; margin-right: 5px;"><path d="M2.5 1a1 1 0 0 0-1 1v1a1 1 0 0 0 1 1H3v9a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V4h.5a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H10a1 1 0 0 0-1-1H7a1 1 0 0 0-1 1H2.5zm3 4a.5.5 0 0 1 .5.5v7a.5.5 0 0 1-1 0v-7a.5.5 0 0 1 .5-.5zM8 5a.5.5 0 0 1 .5.5v7a.5.5 0 0 1-1 0v-7A.5.5 0 0 1 8 5zm3 .5v7a.5.5 0 0 1-1 0v-7a.5.5 0 0 1 1 0z"/></svg>Clear Entire Collection</button></form></div><div class="collection-grid">{collection_images_html}</div></div>{particle_js}</body></html>"""

DETAIL_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Workout Detail: {workout_id}</title><style>body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 0; background-color: #f9f9f9; line-height: 1.6; }} .container {{ max-width: 1000px; margin: 20px auto; padding: 20px; background-color: rgba(255, 255, 255, 0.95); border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); position: relative; z-index: 1; }} h1 {{ color: #333; }} h2 {{ color: #555; border-bottom: 2px solid #eee; padding-bottom: 5px; margin-top: 30px; }} h3 {{ color: #444; text-align: center; margin-bottom: 10px; }} .content {{ display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }} .image-box {{ flex: 1; min-width: 280px; text-align: center; }} .json-box {{ flex: 2; min-width: 400px; }} img {{ border: 2px solid #ccc; border-radius: 4px; background-color: #f0f0f0; }} img.main-img {{ width: 256px; height: 256px; image-rendering: pixelated; }} img.channel-img {{ width: 128px; height: 128px; image-rendering: pixelated; display: block; margin: 10px auto; background-color: #ffffff; }} img.chart-img {{ width: 100%; height: auto; border-color: #eee; }} pre {{ background-color: #black; border: 1px solid #ddd; padding: 10px; border-radius: 4px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word;}} .caption {{ font-style: italic; color: #666; font-size: 0.9em; }} ul {{ padding-left: 20px; }} li {{ margin-bottom: 8px; }} code {{ background-color: #eee; padding: 2px 4px; border-radius: 3px; font-family: monospace; }} .breakdown {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; margin-top: 20px; }} .channel-box {{ border: 1px solid #ddd; border-radius: 6px; padding: 15px; background-color: #fdfdfd; text-align: center; }} .arrow {{ font-size: 24px; font-weight: bold; color: #888; margin: 5px 0; }} .nav-link {{ display: inline-block; margin-bottom: 20px; padding: 8px 15px; background-color: #007aff; color: #fff; text-decoration: none; border-radius: 5px; font-weight: 500; }} .nav-link:hover {{ background-color: #0056b3; }} .guide-box {{ background-color: #fdfdfd; border: 1px solid #eee; border-radius: 8px; padding: 10px 20px; }} .code-block {{ background-color: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: monospace; font-size: 0.9em; }} .code-block pre {{ background-color: inherit; color: inherit; border: none; padding: 0; max-height: none; overflow: visible;}} 

/* AI Summary and Modal Styles */
.ai-summary-box {{ background-color: #fdfdfd; border: 1px solid #eee; border-radius: 8px; padding: 10px 20px; border-top: 4px solid #007aff; }} 
.ai-header {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; }}
.ai-summary-text {{ font-size: 1.1em; font-weight: 500; color: #333; border-left: 4px solid #007aff; padding-left: 10px; margin-top: 10px; padding-top: 5px; padding-bottom: 5px; }}

.modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.4); }}
.modal-content {{ background-color: #fefefe; margin: 10% auto; padding: 20px; border: 1px solid #888; width: 90%; max-width: 800px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }}
.close-btn {{ color: #aaa; float: right; font-size: 28px; font-weight: bold; }}
.close-btn:hover, .close-btn:focus {{ color: #000; text-decoration: none; cursor: pointer; }}
.prompt-code {{ background-color: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: monospace; white-space: pre; }}

</style></head><body><div class="container"><a href="/" class="nav-link">&larr; Back to Workout Gallery</a><h1>Workout Detail: {workout_id}</h1><div class="content"><div class="image-box"><h2>Featured Workout Image</h2><img src="data:image/png;base64,{img_data_base64}" alt="Generated feature image from JSON data" class="main-img"><p class="caption"><b>R:</b> Heart Rate ({hr_min}-{hr_max}bpm)<br><b>G:</b> Calories ({cal_min}-{cal_max}/min)<br><b>B:</b> Speed ({speed_min}-{speed_max}kph)</p></div><div class="json-box"><h2>Original JSON Data</h2><p class="caption">Note the <code>workout_vector</code> field was added automatically.</p><pre><code>{json_data}</code></pre></div></div><div class="ai-summary-box"><div class="ai-header"><h2> AI Summary & Classification ({ai_classification})</h2><div style="display: flex; gap: 10px;">{ai_analysis_button_html}<button id="inspectPromptBtn" class="nav-link" style="margin: 0; cursor: pointer; background-color: #5cb85c;">Inspect LLM Prompt</button></div></div><p>This qualitative summary is generated by an LLM (OpenAI API) using the visual and quantitative data as input, and contextualized by the Vector Search results. **It must be manually generated.**</p><p class="ai-summary-text">{ai_summary}</p><p class="caption">The full analysis prompt was derived from: duration, average HR, max HR, total calories, speed deviation, and the nearest neighbor IDs.</p></div><div class="guide-box" style="margin-top: 30px; border-top: 4px solid #007aff;"><h2> Atlas Vector Search (k-NN)</h2><p>Using a <code>$vectorSearch</code> pipeline in MongoDB with index <code>{vector_index_name}</code> to find the 3 workouts most similar to this one (excluding itself).</p><ul>{ai_neighbors_html}</ul></div><div class="guide-box" style="margin-top: 30px;"><h2>A Radiologist's Guide: How to Read These Images</h2><p>A CNN model learns patterns like a radiologist reads an X-ray:</p><ul><li><strong>Overall Color:</strong> Red=HR, Blue=Speed, Green=Calories. Purplish=High Intensity Run, Yellowish=Cardio/Strength, White=Peak, Dark=Rest/Warmup.</li><li><strong>Texture & Patterns (Top-Left to Bottom-Right):</strong> Top-Left=Warm-up (dark), Bottom-Right=Cool-down (dark). Horizontal Blue Stripes=Intervals, Smooth Gradient=Hill/Pyramid.</li></ul></div><div class="explanation"><h2>Visualizing the Encoding Process</h2><p>How 1D data (64 mins) becomes an 8x8 2D image (read like text).</p><div class="breakdown"><div class="channel-box" style="border-top: 4px solid red;"><h3><span style="color: red;">Red</span> Channel: Heart Rate</h3><img src="data:image/png;base64,{chart_data_hr}" alt="Heart Rate Line Chart" class="chart-img"><div class="arrow">&darr;</div><p class="caption">...is folded into...</p><img src="data:image/png;base64,{img_data_r}" alt="Red channel (grayscale)" class="channel-img"></div><div class="channel-box" style="border-top: 4px solid green;"><h3><span style="color: green;">Green</span> Channel: Calories</h3><img src="data:image/png;base64,{chart_data_cal}" alt="Calories Line Chart" class="chart-img"><div class="arrow">&darr;</div><p class="caption">...is folded into...</p><img src="data:image/png;base64,{img_data_g}" alt="Green channel (grayscale)" class="channel-img"></div><div class="channel-box" style="border-top: 4px solid blue;"><h3><span style="color: blue;">Blue</span> Channel: Speed</h3><img src="data:image/png;base64,{chart_data_speed}" alt="Speed Line Chart" class="chart-img"><div class="arrow">&darr;</div><p class="caption">...is folded into...</p><img src="data:image/png;base64,{img_data_b}" alt="Blue channel (grayscale)" class="channel-img"></div></div></div></div>

<div id="promptModal" class="modal">
  <div class="modal-content">
    <span class="close-btn">&times;</span>
    <h2>Full LLM Prompt Sent to OpenAI</h2>
    <p>This is the exact structured prompt that will be sent to the LLM when you click 'Generate AI Summary'.</p>
    <div class="prompt-code"><pre>{llm_analysis_prompt}</pre></div>
    <p class="caption">The system instruction used to guide the LLM's persona was: 
    "You are a professional Workout Radiologist. Analyze the provided structured workout data and provide a concise, qualitative summary (maximum 3 sentences) from the perspective of a fitness expert. Focus only on the pattern and function of the effort."</p>
  </div>
</div>

<script>
    document.addEventListener("DOMContentLoaded", function() {{
        const modal = document.getElementById("promptModal");
        const btn = document.getElementById("inspectPromptBtn");
        const span = document.getElementsByClassName("close-btn")[0];

        if (btn) {{
            btn.onclick = function() {{
                modal.style.display = "block";
            }}
        }}

        if (span) {{
            span.onclick = function() {{
                modal.style.display = "none";
            }}
        }}

        window.onclick = function(event) {{
            if (event.target == modal) {{
                modal.style.display = "none";
            }}
        }}

        // New: Analysis button logic to handle the async POST request
        const analyzeForm = document.getElementById("analyzeForm");
        const analyzeBtn = document.getElementById("analyzeBtn");

        if (analyzeForm && analyzeBtn) {{
            analyzeForm.addEventListener("submit", function(e) {{
                // Prevent default form submission and start loading state
                e.preventDefault();
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = 'Analyzing...';
                analyzeBtn.style.backgroundColor = '#ff9900';
                
                // Submit via fetch to handle potential network issues
                fetch(analyzeForm.action, {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                }}).then(response => {{
                    if (response.ok) {{
                        // Redirect on success to refresh the page with the new data
                        window.location.reload(); 
                    }} else {{
                        // Handle server-side error response
                        response.json().then(data => {{
                           alert(`LLM Analysis Failed: ${{data.detail}}`);
                        }}).catch(() => {{
                           alert('LLM Analysis Failed. Check console and server logs.');
                        }});
                        
                        // Reset button state
                        analyzeBtn.disabled = false;
                        analyzeBtn.textContent = 'Generate AI Summary';
                        analyzeBtn.style.backgroundColor = '#007aff';
                    }}
                }}).catch(error => {{
                    console.error('Network Error:', error);
                    alert('Network error during analysis.');
                    
                    // Reset button state
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = 'Generate AI Summary';
                    analyzeBtn.style.backgroundColor = '#007aff';
                }});
            }});
        }}
    }});
</script>
{particle_js}</body></html>"""


# ---
# Part 9: ASYNC LLM Functions (OpenAI Only)
# ---

async def call_openai_api(prompt: str) -> str:
  """
  Calls the OpenAI Chat Completions API asynchronously using httpx.
  """
  # System instruction to guide the LLM's persona and output format
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

  # Using a high timeout for the external API call
  async with httpx.AsyncClient(timeout=30.0) as client:
    try:
      logger.info(f"Calling OpenAI API with model {OPENAI_MODEL}...")
      response = await client.post(OPENAI_ENDPOINT, headers=headers, json=payload)
      response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)

      data = response.json()
      summary = data['choices'][0]['message']['content'].strip()
      logger.info("OpenAI API call successful.")
      return summary

    except httpx.HTTPStatusError as e:
      error_msg = f"OpenAI API HTTP error: {e.response.status_code} - {e.response.text}"
      logger.error(error_msg)
      # Raise an exception so the caller knows the API failed
      raise HTTPException(
        status_code=500,
        detail=f"LLM API Error (HTTP {e.response.status_code}): Check API key, quota, or service status."
      )
    except Exception as e:
      error_msg = f"An unexpected error occurred during OpenAI API call: {e}"
      logger.error(error_msg, exc_info=True)
      # Raise an exception so the caller knows the API failed
      raise HTTPException(
        status_code=500,
        detail=f"LLM API Error: Unexpected exception: {e.__class__.__name__}"
      )


def analyze_time_series_features(doc: dict, nearest_neighbors: List[Dict[str, Any]]) -> Tuple[str, str]:
  """
  Analyzes the raw time-series data to extract key features and create a structured LLM prompt.

  Returns: (classification_label, structured_prompt_string)
  """
  ts = doc['time_series']
  hr_data = np.array(ts['heart_rate'])
  cal_data = np.array(ts['calories_per_min'])
  speed_data = np.array(ts['speed_kph'])

  # 1. Quantitative Metrics
  hr_avg = np.mean(hr_data)
  hr_max = np.max(hr_data)
  speed_std = np.std(speed_data)
  cal_total = np.sum(cal_data)

  # 2. Simple Heuristic Classification (based on standard deviation)
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


  # 3. Similarity Context
  neighbors_text = ""
  if nearest_neighbors:
    for i, neighbor in enumerate(nearest_neighbors):
      neighbor_id_suffix = neighbor['_id'].split('_')[-1]
      neighbors_text += f"- Neighbor {i+1}: Workout #{neighbor_id_suffix} (Score: {neighbor['score']:.4f})\n"
  else:
    neighbors_text = "- No close 'Workout Twins' found in the database."

  # 4. Final Structured Prompt
  prompt = f"""
Analyze this structured workout data and provide a concise, qualitative summary (maximum 3 sentences) from the perspective of a **Workout Radiologist**. Focus on the *pattern* and *function* of the effort.

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


# ---
# Part 10: ASYNC FastAPI Routes (Updated for Manual Analysis)
# ---
@app.get('/', response_class=HTMLResponse)
async def show_gallery():
  """Renders the main gallery page."""
  collection_images_html = []
  collection = db.workouts # Access via the non-scoped proxy
  try:
    # Simple find, no scoping filter applied
    workouts_cursor = collection.find({}).sort("_id", ASCENDING)
    workouts = await workouts_cursor.to_list(length=200) # Limit display for performance
  except Exception as e:
    logger.error(f"Error fetching workouts for gallery: {e}")
    workouts = []
    collection_images_html.append(f"<p> Error loading workouts: {e}</p>")

  if not workouts and not collection_images_html:
    collection_images_html.append(f"<p>No workouts found. Click 'Generate'!</p>")

  for doc in workouts:
    try:
      workout_id_suffix_str = doc['_id'].split('_')[-1]
      workout_id_suffix = int(workout_id_suffix_str)
      arrays = generate_workout_viz_arrays(doc, 8) # 8x8 for gallery thumbnails
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
      continue # Skip this item if processing fails

  return HTMLResponse(content=GALLERY_TEMPLATE.format(
    collection_images_html="".join(collection_images_html),
    particle_js=PARTICLE_JS_SCRIPT
  ))


@app.get('/workout/{workout_id}', response_class=HTMLResponse)
async def show_workout_detail(workout_id: int):
  """
  Fetches a specific workout, runs Atlas Vector Search for neighbors,
  and displays the detail page, calculating the LLM prompt if the summary
  hasn't been generated yet (i.e., if placeholders are present).
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

  # Initialize LLM fields from the document
  ai_classification = workout_doc.get("ai_classification", PLACEHOLDER_CLASSIFICATION)
  ai_summary = workout_doc.get("ai_summary", PLACEHOLDER_SUMMARY)
  llm_analysis_prompt = workout_doc.get("llm_analysis_prompt", PLACEHOLDER_PROMPT)
  nearest_neighbors = []
  ai_neighbors_html = ""

  # Flag to check if analysis is pending
  summary_is_pending = (ai_summary == PLACEHOLDER_SUMMARY or ai_classification == PLACEHOLDER_CLASSIFICATION)

  # --- Dynamic Prompt/Neighbor Calculation ---
  if "workout_vector" in workout_doc and isinstance(workout_doc["workout_vector"], list) and len(workout_doc["workout_vector"]) == 192:
    current_vector = workout_doc["workout_vector"]

    # Run Vector Search for neighbors (always needed for the neighbors list display)
    pipeline = [
      {
        "$vectorSearch": {
          "index": VECTOR_INDEX_NAME,
          "path": "workout_vector",
          "queryVector": current_vector,
          "numCandidates": 100,
          "limit": 3,
          "filter": {
            "_id": { "$ne": doc_id }
          }
        }
      },
      { "$project": { "_id": 1, "score": {"$meta": "vectorSearchScore"} } }
    ]

    try:
      neighbors_cursor = collection.aggregate(pipeline)
      nearest_neighbors = await neighbors_cursor.to_list(None)

      # Build HTML for neighbors list
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
            ai_neighbors_html_items.append(f'<li>Neighbor {neighbor_id} (Score: {neighbor_score:.4f})</li>')
        ai_neighbors_html = "".join(ai_neighbors_html_items)
      else:
        ai_neighbors_html = "<p>No other similar workouts found in this collection.</p>"

      # If summary is pending, *calculate* the prompt and classification now for display (for the modal)
      if summary_is_pending:
        ai_classification, llm_analysis_prompt = analyze_time_series_features(workout_doc, nearest_neighbors)

    except OperationFailure as e:
      logger.error(f"OperationFailure during vector search for {doc_id}: {e.details}")
      ai_neighbors_html = f"<p><b> Database error during vector search.</b> Index '{VECTOR_INDEX_NAME}' might be building or failed. <br>Details: {e.details.get('errmsg', str(e))}</p>"
    except Exception as e:
      logger.error(f"Unexpected error during vector search for {doc_id}: {e}")
      ai_neighbors_html = f"<p><b> Application error during vector search:</b> {e}</p>"

  else:
    ai_neighbors_html = "<p>Vector data is missing or malformed for this workout.</p>"

  # --- Prepare Analysis Button HTML ---
  if summary_is_pending:
    ai_analysis_button_html = f"""
      <form id="analyzeForm" action="/workout/{workout_id}/analyze" method="POST" style="margin: 0;">
        <button type="submit" id="analyzeBtn" class="control-btn nav-link" style="margin: 0; cursor: pointer; background-color: #007aff;">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16" style="vertical-align: -2px; margin-right: 5px;"><path d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zm-5.467 4.14C7.02 12.637 7.558 13 8 13c.448 0 .89-.37 1.341-.758.384-.33 1.164-.98 1.956-1.579.529-.396.958-.87 1.253-1.412.308-.567.452-1.217.452-1.921 0-.663-.122-1.284-.367-1.841-.247-.568-.62-1.11-1.12-1.583-.497-.47-1.127-.866-1.87-1.171C9.697 5.093 8.87 4.75 8 4.75c-.878 0-1.688.354-2.457.784-.735.41-1.353.94-1.854 1.572-.497.625-.873 1.342-1.124 2.144-.25.808-.372 1.68-.372 2.616 0 .666.126 1.298.375 1.879.248.568.618 1.107 1.117 1.582.497.47 1.127.865 1.87 1.171z"/><path fill-rule="evenodd" d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14zM8 2A6 6 0 1 1 8 14 6 6 0 0 1 8 2z"/></svg>
          Generate AI Summary
        </button>
      </form>
    """
  else:
    ai_analysis_button_html = '<span style="color: green; font-weight: 600; font-size: 0.9em;">Analysis Complete</span>'

  # Generate visualizations (8x8 arrays used for vector)
  viz_arrays = generate_workout_viz_arrays(workout_doc, 8)
  b64_chart_hr = generate_chart_base64(viz_arrays['raw_hr'], 'Heart Rate (64 mins)', 'red')
  b64_chart_cal = generate_chart_base64(viz_arrays['raw_cal'], 'Calories / min', 'green')
  b64_chart_speed = generate_chart_base64(viz_arrays['raw_speed'], 'Speed (kph)', 'blue')
  b64_combined = encode_pil_image_to_base64(viz_arrays['rgb_combined'], (256, 256), 'RGB')
  b64_r = encode_pil_image_to_base64(viz_arrays['channel_r_2d'], (128, 128), 'L', (255,0,0))
  b64_g = encode_pil_image_to_base64(viz_arrays['channel_g_2d'], (128, 128), 'L', (0,255,0))
  b64_b = encode_pil_image_to_base64(viz_arrays['channel_b_2d'], (128, 128), 'L', (0,0,255))

  # Prepare JSON for display
  doc_for_display = workout_doc.copy()
  if "workout_vector" in doc_for_display and isinstance(doc_for_display["workout_vector"], list):
    vector_preview = str(doc_for_display["workout_vector"][:5])[1:-1]
    doc_for_display["workout_vector"] = f"[{vector_preview}, ... {len(doc_for_display['workout_vector']) - 5} more elements]"

  # Clean up non-visual fields before display
  doc_for_display.pop('experiment_id', None)
  doc_for_display.pop('ai_classification', None)
  doc_for_display.pop('ai_summary', None)
  doc_for_display.pop('llm_analysis_prompt', None)
  json_data_pretty = json.dumps(doc_for_display, indent=2, default=str)

  # Render template
  return HTMLResponse(content=DETAIL_TEMPLATE.format(
    workout_id=workout_id,
    workout_id_full=doc_id,
    img_data_base64=b64_combined,
    json_data=json_data_pretty,
    chart_data_hr=b64_chart_hr, chart_data_cal=b64_chart_cal, chart_data_speed=b64_chart_speed,
    img_data_r=b64_r, img_data_g=b64_g, img_data_b=b64_b,
    hr_min=NORM_BOUNDS["heart_rate"][0], hr_max=NORM_BOUNDS["heart_rate"][1],
    cal_min=NORM_BOUNDS["calories_per_min"][0], cal_max=NORM_BOUNDS["calories_per_min"][1],
    speed_min=NORM_BOUNDS["speed_kph"][0], speed_max=NORM_BOUNDS["speed_kph"][1],
    particle_js=PARTICLE_JS_SCRIPT,
    ai_neighbors_html=ai_neighbors_html,
    vector_index_name=VECTOR_INDEX_NAME,
    ai_classification=ai_classification,
    ai_summary=ai_summary,
    llm_analysis_prompt=llm_analysis_prompt,
    ai_analysis_button_html=ai_analysis_button_html
  ))


@app.post('/workout/{workout_id}/analyze', response_class=RedirectResponse)
async def analyze_workout_and_save(workout_id: int):
  """
  Manually triggers the LLM analysis for a specific workout, updates the document,
  and redirects back to the detail page.
  """
  doc_id = f"workout_6b421a9c_{workout_id}"
  collection = db.workouts

  if not OPENAI_API_KEY:
    error_detail = "OPENAI_API_KEY environment variable is not set. Cannot perform AI workout analysis."
    logger.error(error_detail)
    # This HTTPException will be returned to the JS fetch handler
    raise HTTPException(status_code=503, detail=error_detail)

  try:
    workout_doc = await collection.find_one({"_id": doc_id})
    if not workout_doc or "workout_vector" not in workout_doc:
      raise HTTPException(status_code=404, detail=f"Workout '{doc_id}' not found or missing vector data.")

    current_vector = workout_doc["workout_vector"]

    # 1. Vector Search for Context
    pipeline = [
      {
        "$vectorSearch": {
          "index": VECTOR_INDEX_NAME,
          "path": "workout_vector",
          "queryVector": current_vector,
          "numCandidates": 100,
          "limit": 3,
          "filter": { "_id": { "$ne": doc_id } }
        }
      },
      { "$project": { "_id": 1, "score": {"$meta": "vectorSearchScore"} } }
    ]
    try:
      neighbors_cursor = collection.aggregate(pipeline)
      nearest_neighbors = await neighbors_cursor.to_list(None)
    except OperationFailure:
      nearest_neighbors = []
      logger.warning("Vector search failed during analysis, proceeding without neighbors context.")

    # 2. Generate Structured Prompt
    classification, llm_prompt = analyze_time_series_features(workout_doc, nearest_neighbors)

    # 3. Call the REAL LLM API (This function raises HTTPException on failure)
    ai_summary = await call_openai_api(llm_prompt)

    # 4. Update the document
    update_data = {
      "ai_classification": classification,
      "ai_summary": ai_summary,
      "llm_analysis_prompt": llm_prompt
    }

    await collection.update_one({"_id": doc_id}, {"$set": update_data})
    logger.info(f"Successfully analyzed and updated workout: {doc_id}")

  except HTTPException:
    raise # Re-raise 404/503/500
  except Exception as e:
    logger.error(f"Error during manual AI analysis for {doc_id}: {e}")
    raise HTTPException(status_code=500, detail=f"Server error during analysis: {e.__class__.__name__}")

  # Redirect on successful completion
  return RedirectResponse(url=f'/workout/{workout_id}', status_code=303)


@app.post('/generate')
async def generate_new_workout():
  """
  Creates one new workout, calculates its vector, and inserts the document
  with AI analysis placeholders. LLM is NOT called here.
  """
  async with generation_lock:
    try:
      collection = db.workouts

      # 1. Find the new ID suffix
      latest_workout_cursor = collection.find({}, {"_id": 1}).sort("_id", DESCENDING).limit(1)
      latest_workout = await latest_workout_cursor.to_list(length=1)
      max_id_suffix = int(latest_workout[0]['_id'].split('_')[-1]) if latest_workout else -1

      new_id_suffix = max_id_suffix + 1
      new_doc = create_synthetic_apple_watch_data(new_id_suffix)
      new_doc["workout_vector"] = get_feature_vector(new_doc).tolist()

      # 2. Add placeholders
      new_doc["ai_classification"] = PLACEHOLDER_CLASSIFICATION
      new_doc["ai_summary"] = PLACEHOLDER_SUMMARY
      new_doc["llm_analysis_prompt"] = PLACEHOLDER_PROMPT

      # 3. Insert final document
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
  collection = db.workouts # Use non-scoped proxy
  try:
    delete_result = await collection.delete_many({}) # Simple delete_many
    logger.info(f"Deleted {delete_result.deleted_count} workouts from the collection.")
  except Exception as e:
    logger.error(f"Error clearing collection: {e}")

  return RedirectResponse(url='/', status_code=303)


# ---
# Part 11: The Central Entrypoint (with DB Seeding & Index Check on Startup)
# ---

async def seed_database(collection: CollectionProxy, num_to_seed: int = 20):
  """Seeds the database by creating synthetic workouts with placeholder fields."""
  logger.info(f"Seeding collection with {num_to_seed} workouts...")

  docs_to_insert = []

  # 1. Generate core documents (non-async part)
  for i in range(num_to_seed):
    try:
      doc = create_synthetic_apple_watch_data(i)
      doc["workout_vector"] = get_feature_vector(doc).tolist()
      # Add placeholders instead of calling the LLM
      doc["ai_classification"] = PLACEHOLDER_CLASSIFICATION
      doc["ai_summary"] = PLACEHOLDER_SUMMARY
      doc["llm_analysis_prompt"] = PLACEHOLDER_PROMPT
      docs_to_insert.append(doc)
    except Exception as e:
      logger.error(f"Error creating synthetic data for workout index {i}: {e}")

  if docs_to_insert:
    try:
      # Insert using non-scoped proxy (insert_many handles ignoring duplicates via ordered=False)
      insert_result = await collection.insert_many(docs_to_insert, ordered=False)
      logger.info(f"Attempted to insert {len(docs_to_insert)} workouts. Acknowledged inserts: {len(insert_result.inserted_ids)}.")
    except Exception as e: # Catch potential bulk write errors
      logger.error(f"Error bulk inserting seeded documents: {e}")
  else:
    logger.warning("No documents were generated for seeding.")


@app.on_event("startup")
async def startup_event():
  """Checks/creates index and seeds DB on startup."""
  logger.info("FastAPI app starting up...")
  num_seed_entries = 20 # Initial seed count
  collection = db.workouts # Use non-scoped proxy
  index_manager = collection.index_manager
  index_ready = False
  needs_seeding = False

  MAX_STARTUP_RETRIES = 5
  RETRY_DELAY_SECONDS = 10

  try:
    # Check if seeding is needed (simple count)
    logger.info(f"Checking total document count...")
    count = await collection.count_documents({}, limit=1)
    needs_seeding = (count == 0)
    logger.info(f"Collection is {'empty (needs seeding)' if needs_seeding else 'not empty (skipping seed)'}.")

    # --- Ensure Vector Index (with retries) ---
    for attempt in range(MAX_STARTUP_RETRIES):
      try:
        logger.info(f"Ensuring Atlas Vector Search index '{VECTOR_INDEX_NAME}' exists, is up-to-date, and queryable (Attempt {attempt + 1}/{MAX_STARTUP_RETRIES})...")
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
          logger.warning(f"Attempt {attempt + 1}: create_search_index returned False without raising TimeoutError. Index might still be building or failed.")
          if attempt == MAX_STARTUP_RETRIES - 1:
            logger.critical(f"Vector index '{VECTOR_INDEX_NAME}' did NOT become ready after {MAX_STARTUP_RETRIES} attempts.")
            break

      except (OperationFailure, TimeoutError, AutoReconnect) as e:
        logger.warning(f"Attempt {attempt + 1} failed: Error ensuring search index '{VECTOR_INDEX_NAME}': {getattr(e, 'details', e)}")
        if attempt < MAX_STARTUP_RETRIES - 1:
          logger.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
          await asyncio.sleep(RETRY_DELAY_SECONDS)
        else:
          logger.critical(f"CRITICAL STARTUP ERROR: Failed to ensure search index '{VECTOR_INDEX_NAME}' after {MAX_STARTUP_RETRIES} attempts. Last error: {e}")
          index_ready = False
          break
      except Exception as e:
        logger.critical(f"CRITICAL UNEXPECTED STARTUP ERROR during index check (Attempt {attempt + 1}): {e}", exc_info=True)
        index_ready = False
        break

    # --- Ensure standard _id index ---
    try:
      await index_manager.create_index("_id")
    except Exception as e:
      logger.error(f"Failed to ensure standard _id index: {e}")

    # --- Seed only if needed ---
    if needs_seeding:
      # Seeding inserts documents with placeholders, no LLM call needed here.
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
  logger.info(f"Access the application at http://localhost:{port} (or the mapped port if different)")
  # Use reload=True for development
  uvicorn.run("main:app", host=host, port=port, reload=True)
