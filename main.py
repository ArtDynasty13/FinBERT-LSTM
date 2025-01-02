if __name__ == "__main__":
    from transformers import pipeline
    import pandas as pd
    from tqdm import tqdm  # Progress bar for tracking

    # Load FinBERT model
    pipe = pipeline("text-classification", model="ProsusAI/finbert", device=0)  # Use GPU if available

    # File path to the dataset
    file_path = "./data/91293.CSV"

    # Load and preprocess the data
    data = pd.read_csv(file_path)
    data['PubDate'] = pd.to_datetime(data['PubDate'], errors='coerce')
    data = data.dropna(subset=['PubDate'])
    data['Title'] = data['Title'].astype(str).str.lower().str.strip()

    # Batch processing function
    def get_sentiments_in_batches(titles, batch_size=16):
        sentiments = []
        scores = []
        for i in tqdm(range(0, len(titles), batch_size), desc="Processing batches"):
            batch = titles[i:i + batch_size]
            results = pipe(batch)  # Process the batch
            for result in results:
                sentiments.append(result['label'])
                scores.append(result['score'])
        return sentiments, scores

    # Apply batch sentiment analysis
    batch_size = 16  # Adjust based on available resources
    data['sentiment'], data['score'] = get_sentiments_in_batches(data['Title'].tolist(), batch_size=batch_size)

    # Display the result
    print(data.head())

    # Optionally, save to a new CSV file
    output_file = "./data/91293_sentiment.csv"
    data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
