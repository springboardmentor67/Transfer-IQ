# Integrate With Your React Frontend

This guide explains how to connect your React frontend to the AI Prediction API.

## 1. Start the Backend API

Ensure the FastAPI server is running:

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## 2. API Endpoint Structure

- **URL**: `http://127.0.0.1:8000/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "data": [
      [feat1, feat2, ..., feat9], // Day 1
      [feat1, feat2, ..., feat9], // Day 2
      ...
      [feat1, feat2, ..., feat9]  // Day 7
    ]
  }
  ```

## 3. React Integration Example

Create a new component `PredictionForm.jsx` or add this logic to your player profile page.

```jsx
import React, { useState } from 'react';

const PredictionComponent = () => {
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        
        // Example Data: 7 days of historical data (9 features each)
        // In a real app, you would fetch this from your backend or state
        const inputData = Array(7).fill(0).map(() => 
            Array(9).fill(0).map(() => Math.random() * 100) // Random features for demo
        );

        try {
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ data: inputData }),
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const result = await response.json();
            setPrediction(result.predicted_market_value_eur);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ padding: '20px', border: '1px solid #ccc', borderRadius: '8px' }}>
            <h2>AI Market Value Prediction</h2>
            
            <button 
                onClick={handlePredict} 
                disabled={loading}
                style={{
                    padding: '10px 20px',
                    backgroundColor: '#007bff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: loading ? 'not-allowed' : 'pointer'
                }}
            >
                {loading ? 'Predicting...' : 'Get Live Prediction'}
            </button>

            {error && <p style={{ color: 'red', marginTop: '10px' }}>Error: {error}</p>}

            {prediction !== null && (
                <div style={{ marginTop: '20px' }}>
                    <h3>Predicted Value:</h3>
                    <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#28a745' }}>
                        €{prediction.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </p>
                </div>
            )}
        </div>
    );
};

export default PredictionComponent;
```

## 4. Notes

- **Input Data**: The API expects a sequence of data (e.g., 7 days) with the correct number of features (e.g., 9). Ensure your frontend sends the correct data shape.
- **CORS**: The API is configured to allow requests from any origin (`*`). For production, restrictive CORS settings in `app.py` are recommended.
