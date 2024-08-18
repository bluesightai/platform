# We assume we already dowloaded .tif files from planet or other sources, this is tutorial showing to make it work with mapbox services

#Find out min/max of colors in 16bit tiff:
gdalinfo -stats 1.tif 


# convert files to 8 bit tiff (supported by mapbox) and deal with colors based on the info output, very important to clamp min at 1, not 0 
gdal_translate -ot Byte \
  -b 1 -b 2 -b 3 -b 4 \
  -scale_1 1 9977 1 255 -exponent_1 0.5 \
  -scale_2 2 10783 1 255 -exponent_2 0.5 \
  -scale_3 1 10846 1 255 -exponent_3 0.5 \
  -co COMPRESS=LZW -co PREDICTOR=2 1.tif comp_1.tif  



# NOW WE ARE READY TO UPLOAD TO MAPBOX


# WE SHOULD REMOVE MY TOKEN, ID FROM HERE LATER ON
curl -X POST "https://api.mapbox.com/uploads/v1/szymonzmyslony/credentials?access_token=sk.eyJ1Ijoic3p5bW9uem15c2xvbnkiLCJhIjoiY2x6eWF4cGxsMTJxNDJscHpkZWJqazB4ZCJ9.sUUBnpNhItEWKFAnTuUsbg"
# WE WILL RECEIVE SOME CREDENTIALS:
export AWS_ACCESS_KEY_ID="RECEIVED_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="RECEIVED_SECRET_ACCESS_KEY"
export AWS_SESSION_TOKEN="RECEIVED_SESSION_TOKEN"


# now we upload the file to the provided s3 bucket 
aws s3 cp 8_bit_file.tif s3://RECEIVED_BUCKET/RECEIVED_KEY --region us-east-1

# now we start process of mapbox processing into tile service:
curl -X POST -H "Content-Type: application/json" -H "Cache-Control: no-cache" -d '{
  "url": "https://{BUCKET}.s3.amazonaws.com/{KEY}",
  "tileset": "{USERNAME}.{TILESET_NAME}",
  "name": "{TILESET_DISPLAY_NAME}"
}' 'https://api.mapbox.com/uploads/v1/{USERNAME}?access_token={ACCESS_TOKEN}'



# check the status and if there is no errror - this can be viewed in mapbox dashboard too
curl "https://api.mapbox.com/uploads/v1/szymonzmyslony/UPLOAD_ID?access_token=sk.eyJ1Ijoic3p5bW9uem15c2xvbnkiLCJhIjoiY2x6eWF4cGxsMTJxNDJscHpkZWJqazB4ZCJ9.sUUBnpNhItEWKFAnTuUsbg"













gdal_calc.py -A 1.tif --A_band=1 -B 1.tif --B_band=2 -C 1.tif --C_band=3 \
--outfile=1_8bit_bright.tif \
--calc="numpy.clip(A/2000*255, 0, 255).astype(numpy.uint8)" \
--calc="numpy.clip(B/2000*255, 0, 255).astype(numpy.uint8)" \
--calc="numpy.clip(C/2000*255, 0, 255).astype(numpy.uint8)" \
--type=Byte --co="COMPRESS=LZW" --co="PREDICTOR=2" --co="TILED=YES"