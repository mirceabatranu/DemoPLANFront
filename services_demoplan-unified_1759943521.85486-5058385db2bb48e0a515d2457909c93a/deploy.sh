#!/bin/bash
# DEMOPLAN Unified - Phase 1 Deployment Script with OCR Support
# Single unified agent deployment to Google Cloud Run

set -e

# Configuration
PROJECT_ID="demoplanfrvcxk"
PROJECT_NUMBER="1041867695241"
REGION="europe-west1"
SERVICE_NAME="demoplan-unified"

# OCR Configuration
DOCUMENT_AI_PROCESSOR_ID="f6c4f619b5a674de"
DOCUMENT_AI_LOCATION="eu"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo "========================================="
    echo "🚀 DEMOPLAN UNIFIED - PHASE 1 + OCR"
    echo "Romanian Construction Unified Agent"
    echo "========================================="
}

# Main deployment function
deploy_unified() {
    print_header
    
    print_status "Starting Phase 1 deployment with OCR support..."
    print_status "🏗️ Project: $PROJECT_ID"
    print_status "🌍 Region: $REGION" 
    print_status "🤖 Service: $SERVICE_NAME (Unified Agent + OCR)"
    print_status "🔍 OCR: Document AI Processor $DOCUMENT_AI_PROCESSOR_ID"
    
    # Pre-deployment checks
    print_status "Running pre-deployment checks..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_error "You're not authenticated with gcloud. Please run: gcloud auth login"
        exit 1
    fi
    
    # Set the correct project
    CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
    if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
        print_warning "Current project is $CURRENT_PROJECT, switching to $PROJECT_ID"
        gcloud config set project $PROJECT_ID
    fi
    
    # Check if required files exist
    required_files=("Dockerfile" "requirements.txt" "src/main_api.py" "src/agents/unified_construction_agent.py" "src/services/ocr_service.py")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "Required file not found: $file"
            print_error "Please make sure you're in the DemoPLAN-Unified directory"
            exit 1
        fi
    done
    
    print_success "Pre-deployment checks completed"
    
    # Enable required APIs
    print_status "Enabling required Google Cloud APIs..."
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable run.googleapis.com
    gcloud services enable containerregistry.googleapis.com
    gcloud services enable firestore.googleapis.com
    gcloud services enable documentai.googleapis.com  # NEW: Document AI API
    print_success "APIs enabled (including Document AI)"
    
    # Deploy to Cloud Run using source-based deployment
    print_status "🚀 Deploying Unified Agent with OCR to Cloud Run..."
    print_status "🤖 Phase 1: Single agent with DXF + PDF OCR + Romanian expertise"
    
    gcloud run deploy $SERVICE_NAME \
        --source . \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory 4Gi \
        --cpu 2 \
        --timeout 600 \
        --concurrency 10 \
        --min-instances 0 \
        --max-instances 5 \
        --port 8080 \
        --set-env-vars "\
ENVIRONMENT=production,\
DEBUG=false,\
GCP_PROJECT_ID=$PROJECT_ID,\
GCP_PROJECT_NUMBER=$PROJECT_NUMBER,\
GEMINI_API_KEY=AIzaSyBif3GTOy2uj3A3IncsF4vesDIXAJCfi1M,\
GEMINI_MODEL=gemini-2.0-flash-thinking-exp,\
MIN_CONFIDENCE_FOR_OFFER=75,\
LOG_LEVEL=INFO,\
OCR_ENABLED=true,\
DOCUMENT_AI_PROCESSOR_ID=$DOCUMENT_AI_PROCESSOR_ID,\
DOCUMENT_AI_PROJECT_NUMBER=$PROJECT_NUMBER,\
DOCUMENT_AI_LOCATION=$DOCUMENT_AI_LOCATION,\
OCR_MIN_TEXT_THRESHOLD=100,\
TABLE_EXTRACTION_ENABLED=true,\
HANDWRITING_DETECTION_ENABLED=true,\
ENTITY_EXTRACTION_ENABLED=true,\
OCR_COST_PER_PAGE=0.015,\
OCR_MAX_PAGES_PER_DOC=100"
    
    if [ $? -eq 0 ]; then
        print_success "Phase 1 + OCR deployment completed successfully!"
    else
        print_error "Deployment failed"
        exit 1
    fi
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform=managed --region=$REGION --format='value(status.url)')
    
    print_success "🎉 DEMOPLAN Unified Agent with OCR deployed successfully!"
    echo ""
    echo "========================================="
    echo "📋 PHASE 1 + OCR DEPLOYMENT SUMMARY"
    echo "========================================="
    echo "🌍 Service URL: $SERVICE_URL"
    echo "🤖 Service Name: $SERVICE_NAME"
    echo "🏗️ Region: $REGION"
    echo "💾 Memory: 4GB"
    echo "🖥️ CPU: 2 vCPU"
    echo "⚡ Phase: 1 (Unified Agent + OCR)"
    echo ""
    echo "🔍 OCR Configuration:"
    echo "   Processor ID: $DOCUMENT_AI_PROCESSOR_ID"
    echo "   Location: $DOCUMENT_AI_LOCATION"
    echo "   Processor Type: Document OCR"
    echo "   Cost per page: $0.015 USD"
    echo ""
    echo "📚 API Documentation:"
    echo "   Health Check: $SERVICE_URL/health"
    echo "   Interactive Docs: $SERVICE_URL/docs"
    echo ""
    echo "🔗 Key Endpoints:"
    echo "   Health: GET $SERVICE_URL/health"
    echo "   Start Session: POST $SERVICE_URL/start-session"
    echo "   Upload Files: POST $SERVICE_URL/session/{id}/upload"
    echo "   Chat: POST $SERVICE_URL/session/{id}/chat"
    echo "   Session Status: GET $SERVICE_URL/session/{id}"
    echo ""
    echo "🧪 Testing Commands:"
    echo "   # Health check"
    echo "   curl -f $SERVICE_URL/health"
    echo ""
    echo "   # Start new session"
    echo "   curl -X POST $SERVICE_URL/start-session"
    echo ""
    echo "   # Upload scanned PDF (replace SESSION_ID)"
    echo "   curl -X POST $SERVICE_URL/session/SESSION_ID/upload \\"
    echo "        -F 'files=@scanned_document.pdf'"
    echo ""
    echo "🎯 Phase 1 + OCR Features:"
    echo "   ✅ Unified Construction Agent"
    echo "   ✅ DXF Processing"
    echo "   ✅ PDF Text Extraction"
    echo "   ✅ OCR for Scanned PDFs (NEW)"
    echo "   ✅ Table Extraction (NEW)"
    echo "   ✅ Handwriting Detection (NEW)"
    echo "   ✅ Entity Extraction (NEW)"
    echo "   ✅ Romanian Language Support"
    echo "   ✅ Session Management"
    echo "   ✅ Conversation Interface"
    echo "   ✅ Offer Generation"
    echo ""
    echo "📊 OCR Capabilities:"
    echo "   ✓ Auto-detect scanned documents"
    echo "   ✓ Extract text from images"
    echo "   ✓ Parse tables (costs, specs, materials)"
    echo "   ✓ Detect handwritten notes"
    echo "   ✓ Extract dates, prices, measurements"
    echo "   ✓ Cost tracking per page"
    echo ""
    echo "📄 Next Phases:"
    echo "   Phase 2: Enhanced ML Intelligence"
    echo "   Phase 3: Training Data Pipeline"
    echo "   Phase 4: Continuous Learning"
    echo "========================================="
    
    # Run basic health checks
    print_status "🏥 Running deployment health checks..."
    sleep 20  # Wait for service to be fully ready
    
    # Test health endpoint
    if curl -f -s "$SERVICE_URL/health" > /dev/null; then
        print_success "✅ Health check passed"
        
        # Get detailed health info
        health_response=$(curl -s "$SERVICE_URL/health")
        echo "Health Response: $health_response"
    else
        print_warning "⚠️ Health check failed - service may still be starting"
        print_warning "Wait 1-2 minutes and try: curl $SERVICE_URL/health"
    fi
    
    print_success "🎯 Phase 1 + OCR deployment completed!"
    print_status "📖 Check logs: gcloud run logs read $SERVICE_NAME --region=$REGION"
}

# Run deployment
deploy_unified