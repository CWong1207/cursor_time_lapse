"""
Configuration file for Flask app
Customize your URL settings here
"""
import os

# Server Configuration
HOST = os.getenv('FLASK_HOST', '127.0.0.1')  # '0.0.0.0' = accessible from network, '127.0.0.1' = localhost only
PORT = int(os.getenv('FLASK_PORT', 5000))    # Change this to your desired port (e.g., 8080, 3000, 8000)
DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'  # Set to False for production, True for local development

# Production Domain
# Set this to your actual domain when deployed
CUSTOM_DOMAIN = os.getenv('FLASK_DOMAIN', 'shindo-time-lapse-segmentation.com')

# SSL/HTTPS Configuration (optional - for production)
# SSL_CERT = None  # Path to SSL certificate file
# SSL_KEY = None   # Path to SSL key file
