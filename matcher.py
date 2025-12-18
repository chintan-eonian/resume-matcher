from sentence_transformers import SentenceTransformer
import faiss
import os

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load JD
jd_text = open("data/jd.txt").read()

# Load Resumes
resumes = []
resume_files = os.listdir("data")

for file in resume_files:
    if file.startswith("resume"):
        text = open(f"data/{file}").read()
        resumes.append((file, text))

# Create embeddings
jd_embedding = model.encode([jd_text])
resume_embeddings = model.encode([r[1] for r in resumes])

# Create FAISS index
index = faiss.IndexFlatL2(resume_embeddings.shape[1])
index.add(resume_embeddings)

# Search
D, I = index.search(jd_embedding, k=len(resumes))

# Rank results
print("\n--- Ranked Candidates ---\n")

for rank, idx in enumerate(I[0], start=1):
    name = resumes[idx][0]
    score = (1 / (1 + D[0][rank-1])) * 100
    print(f"{rank}. {name} - Match Score: {round(score,2)}%")
