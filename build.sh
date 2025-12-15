#!/bin/bash
echo "🦁 Building Mufasa Portal and API..."

pip install -r requirements.txt

echo "🌍 Fetching latest frontend files..."
rm -rf frontend-temp
git clone https://github.com/rdhforeclosureconqueror-sys/Mufasa-Real-Assistant frontend-temp

# Copy only the relevant frontend files
cp -r frontend-temp/index.html frontend-temp/assets frontend-temp/portal.css frontend-temp/portal.js frontend-temp/tabs.js .

rm -rf frontend-temp
echo "✅ Frontend merged into API build."
