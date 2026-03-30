import pandas as pd
import numpy as np

def apply_feature_engineering(df):
    """
    Calculates new player metrics: performance_score, goal_ratio, age_factor
    """
    # performance_score = goals + assists
    df['performance_score'] = df['goals'] + df['assists']
    
    # goal_ratio = goals / matches
    df['goal_ratio'] = df['goals'] / df['matches']
    
    # Simple age_factor: Higher value for younger players (peak age ~25-28)
    df['age_factor'] = df['age'].apply(lambda x: 1.0 if 20 <= x <= 28 else 0.5)
    
    return df
