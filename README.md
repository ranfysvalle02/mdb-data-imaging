# **The Workout Radiologist**  

![](smartwatch-visuals.png)

Imagine logging hundreds of workouts, each a unique record of effort and triumph. Now, imagine needing to find that one perfect, brutal interval session you nailed last summer—not by filtering on a name or a date, but by searching for the *feeling* of a specific performance pattern. Traditional databases struggle with this kind of qualitative search, relying solely on simple metadata such as workout titles or durations.  
  
This project, the **FastAPI Workout Radiologist**, transforms that challenge. We move past simple number-crunching to create a **self-contained, high-performance system** that views each workout as a **visual signature**. Built atop a resilient asynchronous Python stack, it leverages the unique capabilities of **MongoDB Atlas Vector Search** to find your *workout twin*—the session that truly *matches the structure and intensity* of your performance fingerprint.  
  
---

## 💡 The Core Innovation: Beyond Traditional Time-Series

We recognize that traditional methods of comparing time-series data, such as **Dynamic Time Warping (DTW)**, are often mathematically intensive and inefficient for massive datasets. Our solution addresses this flaw by transforming complex, one-dimensional time-series logs (Heart Rate, Speed, Calories) into a simple, three-channel $8 \times 8 \times 3$ **visual fingerprint**. This shift allows us to bypass intensive algorithms and use the native power of **MongoDB Atlas Vector Search** and **cosine similarity** to quickly and accurately find your workout twin—the session that truly matches the structural pattern of your effort. We are teaching the database to **"see"** the rhythm of your performance.

---  
  
## **Part 1: The Vision—Turning Time Series into Pixels**  
  
The core breakthrough in this architecture comes from treating *complex, one-dimensional time-series data* like a *simple, two-dimensional image*. Comparing two minute-by-minute heart rate logs is mathematically intensive, often requiring algorithms such as Dynamic Time Warping (DTW). But what if we could let a database index—optimized for embeddings—handle the heavy lifting?  
  
### **Data-to-Image Encoding**  
  
1. **Channel Assignment:**    
   - **Heart Rate** → **Red**    
   - **Calories** → **Green**    
   - **Speed** → **Blue**    
  
2. **Normalization & Folding:**    
   Each metric (e.g., heart rate) contains 64 data points (representing a 64-minute workout). We apply min–max normalization and then *fold* each set of 64 values into an 8×8 array of grayscale values.  
  
3. **The RGB Fingerprint:**    
   Combining the three 8×8 grids results in a single $8 \times 8 \times 3$ **RGB image**. Intervals and ramp patterns appear as recognizable color stripes or gradients.  
  
4. **Vectorization:**    
   Finally, we **flatten the 8×8×3 RGB image** into a single, **192-element numeric vector**—the *workout vector*—that MongoDB Atlas can index for similarity search.  
  
A workout with *pyramid intervals* might show high-contrast horizontal stripes in the **blue** (speed) channel, while a *long slow distance (LSD)* run morphs into a uniform patch of moderate color. This “visual fingerprint” approach lets MongoDB see the **internal structure** of your effort at a glance.  
  
---  
  
### **The Magic of Dimensionality Reduction**  
  
Although we still end with 192 numeric slots, the transformation from temporal data (1D) to visual data (2D) yields a crucial **qualitative** advantage:  
  
- **Folding:** By converting a 64-minute signal into an 8×8 grid, we capture *visual textures* that highlight intervals, ramps, and other patterns.    
- **Cosine Similarity-Ready:** Once in 192D vector form, each workout is quickly comparable via standard embedding metrics.    
- **Lossy But Effective:** We discard some fine-grained detail (e.g., the exact time a sprint began), but preserve the *shape and magnitude* of the effort.  
  
It’s like an X-ray: some detail is lost, but the critical structure stands out.  
  
---  
  
## **Part 2: The Engine Room—Asynchronous Resilience**  
  
To keep the application responsive and fault-tolerant (even during multi-minute index builds), we built the system using **FastAPI** and **Motor** for fully asynchronous I/O.  
  
### **The Necessity of Async for Index Management**  
  
Creating an Atlas Vector Search index isn’t instantaneous; it can take several minutes. A regular synchronous web server would block startup *completely* until the index is done. Instead:  
  
1. **Non-Blocking Polling:**    
   Our custom `AsyncAtlasIndexManager` polls the index status every few seconds via `asyncio.sleep(...)`. During these sleeps, the main event loop remains free to serve requests.  
  
2. **Startup Guarantee:**    
   The `@app.on_event("startup")` hook ensures that the index is *checked* and made queryable—along with handling any needed database seeding—before the application fully declares itself ready.  
  
3. **Graceful Degradation:**    
   If something goes wrong (e.g., network issues, insufficient Atlas tier), the system logs detailed diagnostics but continues to serve “Gallery” pages. You simply lose the advanced similarity feature until the index is healthy.  
  
---  
  
### **The Proxy Pattern: Architectural Elegance**  
  
All the core database logic traverses **DbProxy** or **CollectionProxy**, letting us:  
  
- **Centralize Index Management:** A single method like `collection.index_manager.create_search_index(...)` sets up or updates the vector index.    
- **Simplify Async:** Calls like `db.workouts.find_one(...)` remain natural, but behind the scenes they’re fully asynchronous.    
- **Future Extensibility:** Need logging, caching, or custom error handling? Add it once to the proxy classes and it’s applied everywhere.  
  
---  
  
### **Robust Error Handling: The Five Retry Pattern**  
  
We implement a retry-with-backoff sequence around index creation and readiness checks:  
  
```python  
MAX_STARTUP_RETRIES = 5  
RETRY_DELAY_SECONDS = 10  
  
for attempt in range(MAX_STARTUP_RETRIES):  
    try:  
        # Attempt to create or update index, wait for readiness  
        index_ready = await index_manager.create_search_index(...)  
        if index_ready:  
            break  
    except (OperationFailure, TimeoutError) as e:  
        if attempt < MAX_STARTUP_RETRIES - 1:  
            await asyncio.sleep(RETRY_DELAY_SECONDS)  
        else:  
            logger.critical("Failed after final attempt")  
```  
  
- **Cold Starts:** Atlas clusters may be pausing or spinning up.    
- **Index Failures:** Transient network or resource constraints.    
- **Definition Updates:** Changing from Euclidean to Cosine similarity triggers an index rebuild.    
  
This approach ensures your application *recovers automatically* from these typical production scenarios.  
  
---  
  
## **Part 3: The Discovery—Finding Your Workout Twin**  
  
Once each workout is stored as a 192-element vector, we unlock the power of `$vectorSearch`. For example, a user views *Workout #42* (a steep hill-repeat session) and wants to find a session with *very similar intervals*.  
  
```python  
pipeline = [  
    {  
      "$vectorSearch": {  
        "index": "workout_vector_index",  
        "path": "workout_vector",  
        "queryVector": current_vector,  # 192 elements  
        "numCandidates": 100,  
        "limit": 3,  
        "filter": { "_id": { "$ne": doc_id } }  
      }  
    },  
    { "$project": { "_id": 1, "score": {"$meta": "vectorSearchScore"} } }  
]  
```  
  
MongoDB returns the top matches—based entirely on the *shape* of the data. Maybe *Workout #88* from last month is a near-perfect twin with a similarity score of 0.98. The user can revisit that previous session to see how it compares subjectively.  
  
### **Why This Works: Cosine Similarity in Practice**  
  
MongoDB’s vector search uses **cosine similarity**:  
  
$  
\text{similarity} = \frac{\vec{A} \cdot \vec{B}}{\lVert \vec{A} \rVert\,\lVert \vec{B} \rVert}  
$  
  
Cosine similarity cares about *patterns*, not raw magnitude—meaning if two runners produce the same “intensity curve,” it doesn’t matter if one is a 2-hour marathoner and the other is a 4-hour marathoner; both will match strongly.  
  
This transforms the database from a passive data store into an **intelligent analytics engine**, truly deserving the moniker *Workout Radiologist*, revealing hidden structure in your training data.  
  
---  
  
## **Part 4: The Intelligence Layer—Why We Don’t Let the AI Do the Math**  
  
A highlight of the app is the “Generate AI Summary” feature, returning an expert-like text analysis. However, the crucial math (average heart rate, speed standard deviation, total calories) is done *before* calling the LLM.   
  
### **The Flaw: Trusting Transformers with Totals**  
  
Large Language Models (LLMs) such as GPT-3.5 excel at turning data into coherent text but aren’t guaranteed to do robust, multi-step mathematical calculations reliably. They might:  
  
- Hallucinate (invent a plausible but incorrect stat)    
- Increase cost by requiring more tokens to reason about raw arrays    
- Slow down inference    
  
### **The Solution: Pre-calculated Structured Prompts**  
  
1. **Deterministic Analysis (NumPy)**    
   We compute HR averages, speed standard deviations, and more with 100% accuracy.    
2. **Contextual Grounding (Vector Search)**    
   We gather the top-3 nearest neighbors for each workout.    
3. **Structured Prompt Assembly**    
   These computed facts go into an easily verifiable “data packet.”    
4. **LLM Summarization**    
   Finally, the LLM can add *interpretive flair* to these facts instead of performing the raw math.  
  
### **The Contract: Transparency**  
  
An “Inspect LLM Prompt” button in the UI shows you exactly what the LLM sees—fostering trust. You know your results are anchored by real, deterministic math, not guesswork from the model.  
  
---  
  
## **Appendix A: The Full Technology Stack**  
  
| Layer                        | Technology                   | Purpose                                                              |  
|-----------------------------|------------------------------|----------------------------------------------------------------------|  
| **Web Framework**           | FastAPI + Uvicorn            | Modern, async-compatible HTTP server for low-latency APIs.           |  
| **Database Driver**         | Motor                        | Asynchronous, non-blocking MongoDB driver.                           |  
| **Vector Search**           | MongoDB Atlas Vector Search  | Native indexing with cosine similarity for k-NN queries.             |  
| **LLM Integration**         | OpenAI API (via httpx)       | Summaries and classification text from a GPT-3.5–style model.        |  
| **Data Processing**         | NumPy                        | Fast numerical operations (min–max, std dev).                         |  
| **Image Generation**        | Pillow (PIL)                 | 8×8×3 RGB images plus base64 PNG encoding.                           |  
| **Visualization**           | Matplotlib + Agg backend     | Generates confirmatory line charts for HR, Calories, Speed.          |  
| **Environment**             | python-dotenv                | Secure credential management for `MONGO_URI`, `OPENAI_API_KEY`.      |  
  
### **Critical Dependencies**  
```text  
# motor==3.3.2  
# fastapi==0.104.1  
# httpx==0.25.0  
# numpy==1.26.0  
# pillow==10.1.0  
# matplotlib==3.8.0  
```  
  
---  
  
## **Appendix B: The Vector Index Definition**  
  
Below is the minimal JSON definition for the **Atlas Vector Search index**:  
  
```json  
{  
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
```  
  
- `numDimensions` = 192 matches our 8×8×3 flattened image array.    
- `similarity: "cosine"` is perfect for measuring pattern shape.    
- `"filter"` on `"_id"` lets us exclude the current workout from its own results.  
  
---  
  
## **Appendix C: The Normalization Pipeline**  
  
The `normalize_data()` function ensures consistent pixel ranges:  
  
```python  
def normalize_data(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:  
    clipped_data = np.clip(data, min_val, max_val)  
    range_val = max_val - min_val  
    if range_val == 0:  
        return np.zeros_like(clipped_data, dtype=np.uint8)  
    normalized = (clipped_data - min_val) / range_val  
    return (normalized * 255).astype(np.uint8)  
```  
  
- **Clipping:** Avoids weird spikes (e.g., HR > 200).    
- **Min–Max Scaling:** 0–255 for each metric.    
- **Uint8 Casting:** Perfect for using them as “pixels” in an image.  
  
We define typical bounding ranges:  
  
```python  
NORM_BOUNDS = {  
    "heart_rate": (50, 200),  
    "calories_per_min": (0, 20),  
    "speed_kph": (0, 15)  
}  
```  
  
---  
  
## **Appendix D: The Async Startup Sequence**  
  
During `@app.on_event("startup")`, the application:  
  
1. Checks if the collection is empty and needs seeding.    
2. Tries to create or update the vector index with the correct definition.    
3. Retries up to 5 times if network or index build issues arise.    
4. Seeds 20 synthetic workouts if needed.  
  
The result is an application that can handle cold starts (e.g., an Atlas cluster in sleep mode) gracefully.  
  
---  
  
## **Future Refinements**  
  
1. **Convolutional Autoencoder (CAE):**    
   A trained encoder could learn a smaller, more robust embedding (e.g., 32–64 dimensions) for even stronger similarity results.  
  
2. **User-Weighted Search:**    
   Let the user emphasize Speed vs. Heart Rate vs. Calories on the fly by scaling those pixel intensities before $vectorSearch$.  
  
3. **Detailed Multi-Resolution Encoding:**    
   Split the 64 minutes into sub-blocks (e.g., first half, second half) for greater temporal detail.  
  
4. **Cluster Visualization:**    
   Use t-SNE or UMAP to project high-dimensional vectors onto 2D. Group workouts by structural similarity to quickly see an entire training distribution.  
  
5. **Pre-Workout Suggestions:**    
   Instead of searching *after* a workout has finished, generate recommended patterns for the next workout based on personal or community data.  
  
6. **Integration with Real Wearable APIs:**    
   Swap out synthetic workout data with actual logs from Apple Health, Garmin, or Strava. Immediately see how the system performs on real-world inputs.  
  
---  
  
## **Conclusion: Teaching the Database to "See"**  
  
By **shifting** from numeric time-series to **visual** embeddings, the **FastAPI Workout Radiologist** harnesses MongoDB Atlas Vector Search to find sessions that *feel* the same—regardless of titles or durations. The *deterministic math* for key metrics stays in Python (ensuring accuracy), while the *generative AI* step focuses on producing an insightful textual summary.   
  
In a single stroke, it combines:  
  
- **Asynchronous resilience**    
- **Discrete image-based embedding generation**    
- **Vector-based pattern discovery**    
- **LLM-driven summarization**    
  
It’s a sharp demonstration of modern data engineering—blending robust numeric methods, persistent indexing, and generative intelligence. If you’ve ever wondered how to let your app *see* the meaning behind time-series data, this design points a way forward. And it’s only the beginning—think medical signals, financial markets, or industrial IoT logs. Anywhere there’s a rhythm, a pattern, or a pulse, there’s an opportunity to turn it into a structured image and let advanced similarity search do the rest.  
  
Happy exploring—and may you always find your *workout twin*!  
