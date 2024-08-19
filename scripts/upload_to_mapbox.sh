# We assume we already dowloaded .tif files from planet or other sources, this is tutorial showing to make it work with mapbox services

#Find out min/max of colors in 16bit tiff:
# gdalinfo -stats 1.tif 


# convert files to 8 bit tiff (supported by mapbox) and deal with colors based on the info output, you can use min max for better normalisation, but in genereral 0-3000 is a good range, Planet provides images in BGR so we need to swap channels
# gdal_translate \
#   -ot Byte \
#   -b 3 -b 2 -b 1 \
#   -scale_1 0 3000 0 255 \
#   -scale_2 0 3000 0 255 \
#   -scale_3 0 3000 0 255 \
#   -exponent_1 0.65 \
#   -exponent_2 0.65 \
#   -exponent_3 0.65 \
#   -co COMPRESS=LZW \
#   -co TILED=YES \
#   -co BLOCKXSIZE=256 \
#   -co BLOCKYSIZE=256 \
#   1.tif 2_scaled.tif


# NOW ready to upload to mapbox - just run this and it should work:

# Fetch credentials and pipe the result into DATA
DATA=$(curl -X POST "https://api.mapbox.com/uploads/v1/szymonzmyslony/credentials?access_token=sk.eyJ1Ijoic3p5bW9uem15c2xvbnkiLCJhIjoiY2x6eWF4cGxsMTJxNDJscHpkZWJqazB4ZCJ9.sUUBnpNhItEWKFAnTuUsbg")

# Function to extract values from JSON
get_json_value() {
    echo "$DATA" | grep -o "\"$1\":\"[^\"]*\"" | sed -E 's/"[^"]*":"(.*)"/\1/'
}

# Set AWS credentials
export AWS_ACCESS_KEY_ID=$(get_json_value "accessKeyId")
export AWS_SECRET_ACCESS_KEY=$(get_json_value "secretAccessKey")
export AWS_SESSION_TOKEN=$(get_json_value "sessionToken")

# Set variables
S3_BUCKET=$(get_json_value "bucket")
S3_KEY=$(get_json_value "key")
S3_URL=$(get_json_value "url")
REGION="us-east-1"
FILE_NAME="1.tif"
MAPBOX_USERNAME="szymonzmyslony"
MAPBOX_ACCESS_TOKEN="sk.eyJ1Ijoic3p5bW9uem15c2xvbnkiLCJhIjoiY2x6eWF4cGxsMTJxNDJscHpkZWJqazB4ZCJ9.sUUBnpNhItEWKFAnTuUsbg"
TILESET_NAME="final_tanks"  # Replace with your desired tileset name
TILESET_DISPLAY_NAME="final_tanks"  # Replace with your desired display name

# Upload file to S3
echo "Uploading file to S3..."
aws s3 cp $FILE_NAME s3://$S3_BUCKET/$S3_KEY --region $REGION

# Start Mapbox processing
echo "Initiating Mapbox processing..."
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Cache-Control: no-cache" \
  -d "{
    \"url\": \"$S3_URL\",
    \"tileset\": \"${MAPBOX_USERNAME}.${TILESET_NAME}\",
    \"name\": \"${TILESET_DISPLAY_NAME}\"
  }" \
  "https://api.mapbox.com/uploads/v1/${MAPBOX_USERNAME}?access_token=${MAPBOX_ACCESS_TOKEN}"

echo "Process completed."


