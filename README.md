# Programming Language Detector

A full-stack application that detects programming languages from code snippets.

## Features

- Modern, responsive UI built with React and Material-UI
- FastAPI backend serving ML model predictions
- Real-time language detection
- Support for multiple programming languages

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI server
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Main React component
│   │   └── ...
│   └── package.json        # Node.js dependencies
├── vectorizer.pkl          # ML model files
├── language_classifier.pkl
└── label_encoder.pkl
```

## Starting the Application

You need to start both the backend and frontend servers in separate terminal windows.

### 1. Start the Backend Server (Terminal 1)
```bash
# Navigate to backend directory
cd backend

# Start the FastAPI server
uvicorn main:app --reload --port 9000
```

### 2. Start the Frontend Server (Terminal 2)
```bash
# Navigate to frontend directory
cd frontend

# Start the React development server
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:9000

## Usage

1. Open the application in your web browser
2. Paste your code snippet into the text area
3. Click "Detect Language"
4. The predicted programming language will be displayed below

## API Endpoints

- `POST /predict`
  - Request body: `{ "code": "your code snippet here" }`
  - Response: `{ "language": "detected language", "confidence": "confidence level" }`

## Tech Stack
- Frontend: React + TypeScript + Material-UI
## Technologies Used

- Frontend:
  - React with TypeScript
  - Material-UI
  - Axios

- Backend:
  - FastAPI
  - scikit-learn
  - XGBoost

## License

MIT 