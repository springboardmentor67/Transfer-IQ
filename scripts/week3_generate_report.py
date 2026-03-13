import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Loads sentiment features and final dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sentiment_path = os.path.join(base_dir, "data", "processed", "player_sentiment_features.csv")
    final_data_path = os.path.join(base_dir, "data", "processed", "final_dataset.csv")
    
    if not os.path.exists(sentiment_path):
        logging.error(f"Sentiment data not found: {sentiment_path}")
        return None, None
    if not os.path.exists(final_data_path):
        logging.error(f"Final dataset not found: {final_data_path}")
        return None, None
        
    sent_df = pd.read_csv(sentiment_path)
    final_df = pd.read_csv(final_data_path)
    
    return sent_df, final_df

def merge_datasets(sent_df, final_df):
    """Merges sentiment data with market value data."""
    # Ensure correct merge keys
    if 'player_name' not in sent_df.columns or 'player_name' not in final_df.columns:
        logging.error("Missing 'player_name' column for merge.")
        return None

    # Check if market_value_eur exists
    if 'market_value_eur' not in final_df.columns:
        logging.warning("market_value_eur not found in final dataset. Proceeding without correlation plots.")
        merged_df = sent_df # Just use sentiment data
    else:
        merged_df = pd.merge(sent_df, final_df[['player_name', 'market_value_eur']], on='player_name', how='inner')
        logging.info(f"Merged {len(merged_df)} players for correlation analysis.")
        
    return merged_df

def create_title_page(pdf, num_players):
    """Creates a title page for the PDF report."""
    plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    plt.text(0.5, 0.6, "Sentiment Analysis Report\n(Week 3-4)", ha='center', va='center', fontsize=24, weight='bold')
    plt.text(0.5, 0.4, f"Analyzed {num_players} Players", ha='center', va='center', fontsize=16)
    pdf.savefig()
    plt.close()

def create_text_page(pdf, title, lines):
    """Creates a simple text page."""
    plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    plt.text(0.1, 0.9, title, ha='left', va='top', fontsize=18, weight='bold')
    
    y_pos = 0.8
    for line in lines:
        plt.text(0.1, y_pos, line, ha='left', va='top', fontsize=12)
        y_pos -= 0.05
        
    pdf.savefig()
    plt.close()

def generate_pdf_report(df, output_dir):
    """Generates a PDF report with visualizations and insights."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "sentiment_analysis_report.pdf")
    
    sns.set_theme(style="whitegrid")
    
    top_positive = df.sort_values('vader_compound_mean', ascending=False).head(5)
    top_impactful = df.sort_values('sentiment_impact_signed_sum', ascending=False).head(5)
    
    with PdfPages(report_path) as pdf:
        # 1. Title Page
        create_title_page(pdf, len(df))
        
        # 2. Key Findings Page
        findings = []
        findings.append("Top 5 Most Positive Players:")
        for _, row in top_positive.iterrows():
            findings.append(f"- {row['player_name']}: {row['vader_compound_mean']:.2f}")
        
        findings.append("\nTop 5 High Impact Players (Sentiment * Popularity):")
        for _, row in top_impactful.iterrows():
            findings.append(f"- {row['player_name']}: {row['sentiment_impact_signed_sum']:.2f}")
            
        create_text_page(pdf, "Key Findings", findings)
        
        # 3. Sentiment Distribution Plot
        plt.figure(figsize=(10, 6))
        sns.histplot(df['vader_compound_mean'], bins=20, kde=True, color='skyblue')
        plt.title('Distribution of Average Sentiment per Player')
        plt.xlabel('Average VADER Compound Score')
        plt.ylabel('Count')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        if 'market_value_eur' in df.columns:
            # 4. Impact vs Market Value
            if 'sentiment_impact_signed_sum' in df.columns:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df, x='sentiment_impact_signed_sum', y='market_value_eur', alpha=0.7)
                plt.title('Total Sentiment Impact vs. Market Value')
                plt.xlabel('Total Sentiment Impact')
                plt.ylabel('Market Value (EUR)')
                plt.tight_layout()
                pdf.savefig()
                plt.close()

            # 5. Popularity vs Market Value
            if 'popularity_score_sum' in df.columns:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df, x='popularity_score_sum', y='market_value_eur', color='orange', alpha=0.7)
                plt.title('Total Popularity Score vs. Market Value')
                plt.xlabel('Total Popularity Score')
                plt.ylabel('Market Value (EUR)')
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                
    logging.info(f"PDF Report generated at {report_path}")

def main():
    sent_df, final_df = load_data()
    if sent_df is not None and final_df is not None:
        merged_df = merge_datasets(sent_df, final_df)
        if merged_df is not None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, "reports")
            
            generate_pdf_report(merged_df, output_dir)

if __name__ == "__main__":
    main()
