# Face Analyzer

A web application that analyzes faces from LinkedIn profile images to determine ethnicity, age, and gender using deep learning.

## Features

- Upload CSV file with LinkedIn profile URLs
- Automatic face detection and analysis
- Displays:
  - Primary Race
  - Sub Ethnicity
  - Age
  - Gender
- Clean and modern UI
- Progress tracking
- Batch processing support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LETME2X/race_detection
cd face-analyzer
```

2. Create a virtual environment and activate it:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Upload a CSV file with the following format:
```csv
name,profile_url
John Doe,https://example.com/profile.jpg
```

## CSV File Format

The CSV file should contain two columns:
- `name`: The person's name
- `profile_url`: Direct URL to their profile image

## Technologies Used

- Backend:
  - FastAPI
  - DeepFace
  - SQLite
  - Python 3.8+
- Frontend:
  - HTML5
  - CSS3
  - JavaScript (Vanilla)

## Project Structure

```
face-analyzer/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── static/            # Static files
│   └── index.html     # Frontend UI
└── profile_images/    # Uploaded images storage
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
