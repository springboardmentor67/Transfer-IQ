import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
    def analyze_text(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {
                "vader_compound": 0.0,
                "textblob_polarity": 0.0,
                "textblob_subjectivity": 0.0
            }
        
        vader_scores = self.vader.polarity_scores(text)
        blob = TextBlob(text)
        
        return {
            "vader_compound": vader_scores["compound"],
            "vader_pos": vader_scores["pos"],
            "vader_neg": vader_scores["neg"],
            "vader_neu": vader_scores["neu"],
            "textblob_polarity": blob.sentiment.polarity,
            "textblob_subjectivity": blob.sentiment.subjectivity
        }
    
    def analyze_tweets(self, tweets: list) -> dict:
        if not tweets:
            return {}
        
        results = [self.analyze_text(t) for t in tweets]
        df = pd.DataFrame(results)
        
        mean_scores = df.mean().to_dict()
        mean_scores["total_analyzed"] = len(tweets)
        return mean_scores

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    print("Testing Sentiment Analyzer:")
    sample = ["What a fantastic goal by Messi!", "He looks injured and played terribly."]
    res = analyzer.analyze_tweets(sample)
    print(res)
