from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import torch
import numpy as np
from PIL import Image
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import Polygon, mapping, box, Point
from shapely.prepared import prep
import pandas as pd
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import json
import requests
from io import BytesIO
import mercantile
from rasterio.merge import merge
from rasterio.warp import transform_bounds, transform
from pyproj import Transformer
import math
import time
import logging
import sys

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'tif', 'tiff'}
MODEL_PATH = 'model/segformer_model.pth'
IRRADIANCE_DATA_PATH = 'data/irradiance_data.csv'
MAPBOX_TOKEN = 'YOUR_MAPBOX_TOKEN'  # Replace with your Mapbox token
TILE_SIZE = 256
PATCH_SIZE = 512  # Size of patches to process

# Karnataka boundary coordinates (approximate bounding box)
KARNATAKA_BOUNDS = {
    'north': 18.4,  # Northern boundary
    'south': 11.5,  # Southern boundary
    'west': 74.0,   # Western boundary
    'east': 78.6    # Eastern boundary
}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Ensure upload and results directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model = model.to(device)
    return model, device

def download_satellite_tile(x, y, z):
    """Download satellite tile from Google"""
    url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    return None

def get_tiles_for_bounds(bounds, zoom):
    """Get all tile coordinates that cover the given bounds at specified zoom level"""
    west, south, east, north = bounds
    tiles = list(mercantile.tiles(west, south, east, north, zoom))
    return tiles

def create_mosaic_from_tiles(tiles, zoom):
    """Create a mosaic image from multiple map tiles"""
    # Calculate the size of the final mosaic
    min_x = min(tile.x for tile in tiles)
    max_x = max(tile.x for tile in tiles)
    min_y = min(tile.y for tile in tiles)
    max_y = max(tile.y for tile in tiles)
    
    width = (max_x - min_x + 1) * TILE_SIZE
    height = (max_y - min_y + 1) * TILE_SIZE
    
    mosaic = Image.new('RGB', (width, height))
    
    for tile in tiles:
        img = download_satellite_tile(tile.x, tile.y, zoom)
        if img:
            x_pos = (tile.x - min_x) * TILE_SIZE
            y_pos = (tile.y - min_y) * TILE_SIZE
            mosaic.paste(img, (x_pos, y_pos))
    
    return mosaic

def process_image_patches(image, processor, model, device, patch_size=512, overlap=64):
    """Process large images in patches"""
    width, height = image.size
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(0, height, patch_size - overlap):
        for x in range(0, width, patch_size - overlap):
            # Extract patch
            patch = image.crop((x, y, min(x + patch_size, width), min(y + patch_size, height)))
            
            # Process patch
            inputs = processor(images=patch, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**{k: v.to(device) for k, v in inputs.items()})
                patch_mask = outputs.logits.argmax(dim=1)[0].cpu().numpy()
            
            # Resize patch mask to original size if needed
            if patch_mask.shape != (patch_size, patch_size):
                patch_mask = Image.fromarray(patch_mask.astype(np.uint8))
                patch_mask = patch_mask.resize(patch.size)
                patch_mask = np.array(patch_mask)
            
            # Insert patch mask into full mask
            mask[y:min(y + patch_size, height), x:min(x + patch_size, width)] = patch_mask
    
    return mask

def save_input_visualization(image_array, filename):
    """Save the input image for visualization"""
    # Ensure the image is in uint8 format
    if image_array.dtype != np.uint8:
        image_array = ((image_array - image_array.min()) * (255 / (image_array.max() - image_array.min()))).astype(np.uint8)
    
    # Create PIL image
    input_image = Image.fromarray(image_array)
    
    # Save the image
    image_path = os.path.join(RESULTS_FOLDER, filename)
    input_image.save(image_path)
    
    return image_path

def process_image(image_path, processor, model, device):
    """Process a GeoTIFF image"""
    try:
        logger.info(f"Opening image: {image_path}")
        with rasterio.open(image_path) as src:
            # Read image data
            image = src.read()
            original_transform = src.transform
            crs = src.crs
            
            logger.info(f"Image shape: {image.shape}")
            logger.info(f"Original transform: {original_transform}")
            logger.info(f"CRS: {crs}")
            
            # Calculate target resolution (0.6 meters)
            TARGET_RESOLUTION = 0.6  # meters
            
            # Calculate scaling factors
            if crs and crs.is_geographic:
                # Convert degrees to approximate meters (at equator, 1 degree ≈ 111,320 meters)
                current_res_x = abs(original_transform[0]) * 111320
                current_res_y = abs(original_transform[4]) * 111320
            else:
                current_res_x = abs(original_transform[0])
                current_res_y = abs(original_transform[4])
            
            logger.info(f"Current resolution: {current_res_x}m x {current_res_y}m")
            
            scale_factor_x = current_res_x / TARGET_RESOLUTION
            scale_factor_y = current_res_y / TARGET_RESOLUTION
            
            logger.info(f"Scale factors: {scale_factor_x} x {scale_factor_y}")
            
            # Calculate new dimensions
            new_width = max(int(image.shape[2] * scale_factor_x), 1)
            new_height = max(int(image.shape[1] * scale_factor_y), 1)
            
            logger.info(f"New dimensions: {new_width} x {new_height}")
            
            # Create new transform for the resampled image
            new_transform = rasterio.transform.from_bounds(
                src.bounds.left, src.bounds.bottom,
                src.bounds.right, src.bounds.top,
                new_width, new_height
            )
            
            logger.info(f"New transform: {new_transform}")
            
            # Resample image to 0.6m resolution
            resampled_image = np.zeros((image.shape[0], new_height, new_width), dtype=image.dtype)
            for band in range(image.shape[0]):
                resampled_image[band] = rasterio.warp.reproject(
                    source=image[band],
                    destination=resampled_image[band],
                    src_transform=original_transform,
                    src_crs=crs,
                    dst_transform=new_transform,
                    dst_crs=crs,
                    resampling=rasterio.enums.Resampling.bilinear
                )[0]
            
            logger.info(f"Resampled image shape: {resampled_image.shape}")
            
            # Convert to RGB if necessary
            if resampled_image.shape[0] > 3:
                resampled_image = resampled_image[:3]  # Take first 3 channels
            elif resampled_image.shape[0] == 1:  # If single channel
                resampled_image = np.stack([resampled_image[0]] * 3, axis=0)  # Convert to 3 channels
            
            # Convert to HWC format (height, width, channels)
            image = np.moveaxis(resampled_image, 0, -1)
            
            # Ensure values are in uint8 range
            if image.dtype != np.uint8:
                image = ((image - image.min()) * (255 / (image.max() - image.min()))).astype(np.uint8)
            
            logger.info(f"Final image shape: {image.shape}")
            
            # Save original image for visualization
            input_filename = os.path.basename(image_path).rsplit('.', 1)[0] + '_input.png'
            save_input_visualization(image, input_filename)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            logger.info("Processing image through model")
            # Process through model
            inputs = processor(images=pil_image, return_tensors="pt", do_rescale=True)
            
            with torch.no_grad():
                outputs = model(**{k: v.to(device) for k, v in inputs.items()})
                logits = outputs.logits
                mask = logits.argmax(dim=1)[0].cpu().numpy()
            
            logger.info(f"Generated mask shape: {mask.shape}")
            
            return mask, new_transform, crs, input_filename
            
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        logger.error(f"Stack trace: {sys.exc_info()}")
        raise

def mask_to_geojson(mask, transform, crs):
    """Convert mask to GeoJSON with proper georeferencing"""
    # Get shapes from mask
    shapes_generator = shapes(mask.astype(np.int16), mask=(mask > 0), transform=transform)
    
    # Convert to GeoJSON features
    features = []
    for geometry, value in shapes_generator:
        # Convert geometry to Shapely polygon
        polygon = Polygon(geometry['coordinates'][0])
        
        # Calculate area in square meters
        if crs:
            # Create GeoDataFrame with the correct CRS
            gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs=crs)
            # Convert to UTM for accurate area calculation
            utm_crs = get_utm_crs(polygon.centroid.x, polygon.centroid.y)
            gdf_utm = gdf.to_crs(utm_crs)
            area = gdf_utm.geometry.area.values[0]
        else:
            # For images without CRS, use pixel dimensions
            # Assuming 0.6m resolution per pixel
            area = polygon.area * (0.6 * 0.6)
        
        feature = {
            "type": "Feature",
            "geometry": mapping(polygon),
            "properties": {
                "area": float(area)  # Area in square meters
            }
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

def get_utm_crs(lon, lat):
    """Get the appropriate UTM CRS for a given longitude and latitude"""
    zone_number = int((lon + 180) / 6) + 1
    return f'EPSG:326{zone_number:02d}' if lat >= 0 else f'EPSG:327{zone_number:02d}'

def load_irradiance_data():
    """Load irradiance data from CSV file"""
    try:
        # Define the column names from the CSV structure
        columns = ['PARAMETER', 'YEAR', 'LAT', 'LON', 'JAN', 'FEB', 'MAR', 'APR', 
                  'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANN']
        
        # Skip the header rows (first 8 lines) and use explicit column names
        df = pd.read_csv(IRRADIANCE_DATA_PATH, skiprows=8, names=columns)
        
        # Debug info before filtering
        logger.info(f"Total rows loaded: {len(df)}")
        logger.info(f"Years in data: {df['YEAR'].unique()}")
        logger.info(f"Sample of first few rows:\n{df.head()}")
        
        # Convert YEAR to numeric, handling any non-numeric values
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
        
        # Filter for most recent year (2021)
        year_data = df[df['YEAR'] == 2021]
        
        if len(year_data) == 0:
            # If no 2021 data, try 2020
            logger.info("No data found for 2021, trying 2020...")
            year_data = df[df['YEAR'] == 2020]
        
        if len(year_data) == 0:
            logger.error("No valid data found for either 2020 or 2021")
            return None
            
        df = year_data  # Use the filtered data
        logger.info(f"Using data for year: {int(df['YEAR'].iloc[0])}")
        
        # Convert all numeric columns to float, handling any non-numeric values
        numeric_columns = ['LAT', 'LON', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                         'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANN']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert average daily W/m² to annual kWh/m²/year
        # ANN is average daily irradiance in W/m²
        # For annual energy: multiply by 24 hours/day and 365 days/year, divide by 1000 for kWh
        df['annual_irradiance'] = df['ANN'] * 24 * 365 / 1000
        
        logger.info(f"Loaded irradiance data: {len(df)} grid points")
        logger.info(f"Average daily irradiance range: {df['ANN'].min():.1f} to {df['ANN'].max():.1f} W/m²")
        logger.info(f"Annual irradiance range: {df['annual_irradiance'].min():.0f} to {df['annual_irradiance'].max():.0f} kWh/m²/year")
        logger.info(f"Latitude range: {df['LAT'].min():.1f}°N to {df['LAT'].max():.1f}°N")
        logger.info(f"Longitude range: {df['LON'].min():.1f}°E to {df['LON'].max():.1f}°E")
        
        # Verify the data covers Karnataka region
        karnataka_data = df[
            (df['LAT'] >= KARNATAKA_BOUNDS['south']) &
            (df['LAT'] <= KARNATAKA_BOUNDS['north']) &
            (df['LON'] >= KARNATAKA_BOUNDS['west']) &
            (df['LON'] <= KARNATAKA_BOUNDS['east'])
        ]
        
        if len(karnataka_data) == 0:
            logger.error("No irradiance data points found within Karnataka boundaries")
            logger.error(f"Looking for data between: {KARNATAKA_BOUNDS['south']}-{KARNATAKA_BOUNDS['north']}°N, {KARNATAKA_BOUNDS['west']}-{KARNATAKA_BOUNDS['east']}°E")
            return None
            
        logger.info(f"Found {len(karnataka_data)} data points within Karnataka boundaries")
        logger.info(f"Karnataka average daily irradiance range: {karnataka_data['ANN'].min():.1f} to {karnataka_data['ANN'].max():.1f} W/m²")
        logger.info(f"Karnataka annual irradiance range: {karnataka_data['annual_irradiance'].min():.0f} to {karnataka_data['annual_irradiance'].max():.0f} kWh/m²/year")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading irradiance data: {str(e)}")
        logger.error("CSV structure:")
        try:
            # Try to read and log the first few lines to help with debugging
            with open(IRRADIANCE_DATA_PATH, 'r') as f:
                header = ''.join([next(f) for _ in range(10)])
            logger.error(f"First 10 lines:\n{header}")
        except Exception as e2:
            logger.error(f"Could not read CSV header: {str(e2)}")
        return None

def get_irradiance_for_location(lat, lon, irradiance_data):
    """Get irradiance value for a specific location using bilinear interpolation"""
    if irradiance_data is None:
        # Return default value if data is not available
        return 1825  # Default annual irradiance for Karnataka
    
    try:
        # Find the four nearest grid points
        lat_low = math.floor(lat * 2) / 2
        lat_high = math.ceil(lat * 2) / 2
        lon_low = math.floor(lon * 2) / 2
        lon_high = math.ceil(lon * 2) / 2
        
        # Get values for the four corners
        corners = irradiance_data[
            (irradiance_data['LAT'].isin([lat_low, lat_high])) &
            (irradiance_data['LON'].isin([lon_low, lon_high]))
        ]
        
        if len(corners) < 4:
            # If we don't have all four corners, return the nearest point
            nearest = irradiance_data.iloc[
                ((irradiance_data['LAT'] - lat)**2 + 
                 (irradiance_data['LON'] - lon)**2).argmin()
            ]
            return nearest['annual_irradiance']
        
        # Perform bilinear interpolation
        Q11 = corners[(corners['LAT'] == lat_low) & (corners['LON'] == lon_low)]['annual_irradiance'].iloc[0]
        Q12 = corners[(corners['LAT'] == lat_low) & (corners['LON'] == lon_high)]['annual_irradiance'].iloc[0]
        Q21 = corners[(corners['LAT'] == lat_high) & (corners['LON'] == lon_low)]['annual_irradiance'].iloc[0]
        Q22 = corners[(corners['LAT'] == lat_high) & (corners['LON'] == lon_high)]['annual_irradiance'].iloc[0]
        
        # Interpolation weights
        x = (lon - lon_low) / (lon_high - lon_low)
        y = (lat - lat_low) / (lat_high - lat_low)
        
        # Bilinear interpolation formula
        irradiance = (
            Q11 * (1 - x) * (1 - y) +
            Q21 * (1 - x) * y +
            Q12 * x * (1 - y) +
            Q22 * x * y
        )
        
        return irradiance
        
    except Exception as e:
        logger.error(f"Error getting irradiance data: {str(e)}")
        return 1825  # Default value

def calculate_solar_potential(geojson_data):
    PANEL_EFFICIENCY = 0.18
    STANDARD_TEST_IRRADIANCE = 1000  # W/m²
    
    # Load irradiance data
    irradiance_data = load_irradiance_data()
    
    total_area = 0
    total_power = 0
    total_generation = 0
    
    for feature in geojson_data['features']:
        try:
            area = feature['properties']['area']
            if area > 0:
                # Get center point of the feature for irradiance lookup
                polygon = Polygon(feature['geometry']['coordinates'][0])
                centroid = polygon.centroid
                annual_irradiance = get_irradiance_for_location(
                    centroid.y, centroid.x, 
                    irradiance_data
                )
                
                total_area += area
                instantaneous_power = area * STANDARD_TEST_IRRADIANCE * PANEL_EFFICIENCY
                annual_generation = area * annual_irradiance * PANEL_EFFICIENCY
                total_power += instantaneous_power
                total_generation += annual_generation
                
                feature['properties'].update({
                    'instantaneous_power': float(instantaneous_power),
                    'annual_generation': float(annual_generation),
                    'annual_irradiance': float(annual_irradiance)
                })
        except Exception as e:
            logger.error(f"Error processing feature: {str(e)}")
            continue
    
    geojson_data['metadata'] = {
        'total_area': float(total_area),
        'total_annual_generation': float(total_generation),
        'total_instantaneous_power': float(total_power),
        'panel_efficiency': PANEL_EFFICIENCY,
        'panel_type': 'Monocrystalline PERC',
        'panel_lifespan': 25
    }
    
    return geojson_data

def save_mask_visualization(mask, filename):
    """Save the predicted mask as a grayscale image"""
    # Convert binary mask to 0-255 range
    vis_mask = mask.astype(np.uint8) * 255
    
    # Create PIL image
    mask_image = Image.fromarray(vis_mask)
    
    # Save the image
    mask_path = os.path.join(RESULTS_FOLDER, filename)
    mask_image.save(mask_path)
    
    return mask_path

def format_power(power_watts):
    """Format power in appropriate units (W, kW, or MW)"""
    if power_watts >= 1_000_000:
        return f"{power_watts/1_000_000:.2f} MW"
    elif power_watts >= 1_000:
        return f"{power_watts/1_000:.2f} kW"
    else:
        return f"{power_watts:.0f} W"

def is_within_karnataka(north, south, east, west):
    """Check if the given coordinates fall within Karnataka's boundaries"""
    # Check if the region's bounds intersect with Karnataka's bounds
    if (south > KARNATAKA_BOUNDS['north'] or 
        north < KARNATAKA_BOUNDS['south'] or 
        west > KARNATAKA_BOUNDS['east'] or 
        east < KARNATAKA_BOUNDS['west']):
        return False
    return True

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/process', methods=['POST'])
def process():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved uploaded file to: {filepath}")

        # Load model and processor
        model, device = load_model()
        processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
        
        # Process image
        logger.info("Processing image")
        mask, transform, crs, input_filename = process_image(filepath, processor, model, device)
        
        # Save mask visualization
        mask_filename = os.path.splitext(filename)[0] + '_mask.png'
        mask_path = save_mask_visualization(mask, mask_filename)
        logger.info(f"Saved mask visualization to: {mask_path}")
        
        # Convert mask to GeoJSON
        logger.info("Converting mask to GeoJSON")
        geojson_data = mask_to_geojson(mask, transform, crs)
        
        # Calculate solar potential
        logger.info("Calculating solar potential")
        geojson_with_solar = calculate_solar_potential(geojson_data)
        
        # Calculate totals
        total_area = sum(feature['properties']['area'] for feature in geojson_with_solar['features'])
        total_generation = geojson_with_solar['metadata']['total_annual_generation']
        total_power = geojson_with_solar['metadata']['total_instantaneous_power']
        
        logger.info(f"Total area: {total_area:.2f} m²")
        logger.info(f"Total generation: {total_generation:.2f} kWh")
        logger.info(f"Total power: {total_power:.2f} W")
        
        # Format area for display
        formatted_area = f"{total_area:.1f} m²"
        if total_area > 10000:
            formatted_area = f"{total_area/10000:.2f} hectares ({total_area:.1f} m²)"
        
        # Ensure panel info is properly structured
        panel_info = {
            'type': geojson_with_solar['metadata'].get('panel_type', 'Monocrystalline PERC'),
            'efficiency': geojson_with_solar['metadata'].get('panel_efficiency', 0.18) * 100,
            'lifespan': geojson_with_solar['metadata'].get('panel_lifespan', 25)
        }
        
        return jsonify({
            'success': True,
            'result_file': os.path.splitext(filename)[0] + '_result.geojson',
            'mask_file': mask_filename,
            'input_file': input_filename,
            'total_area': formatted_area,
            'total_area_m2': total_area,
            'total_generation': total_generation,
            'total_power': format_power(total_power),
            'panel_info': panel_info,
            'geojson': geojson_with_solar
        })

    except Exception as e:
        logger.error(f"Error in process endpoint: {str(e)}")
        logger.error(f"Stack trace: {sys.exc_info()}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_roi', methods=['POST'])
def process_roi():
    try:
        data = request.get_json()
        if not all(k in data for k in ['north', 'south', 'east', 'west']):
            return jsonify({'error': 'Missing coordinates'}), 400

        # Check if the region is within Karnataka
        if not is_within_karnataka(data['north'], data['south'], data['east'], data['west']):
            return jsonify({
                'error': 'Selected region is outside Karnataka. Please select an area within Karnataka as solar irradiance data is only available for this region.'
            }), 400

        # Get optimal zoom level for the area
        zoom = 19  # Maximum zoom for highest resolution
        
        # Get tiles covering the area
        bounds = (data['west'], data['south'], data['east'], data['north'])
        tiles = get_tiles_for_bounds(bounds, zoom)
        
        # Create mosaic from tiles
        mosaic = create_mosaic_from_tiles(tiles, zoom)
        
        if mosaic is None or mosaic.size[0] == 0 or mosaic.size[1] == 0:
            return jsonify({'error': 'Failed to create image mosaic'}), 500
        
        # Save input visualization
        input_filename = f"roi_{int(time.time())}_input.png"
        input_path = os.path.join(RESULTS_FOLDER, input_filename)
        mosaic.save(input_path)
        
        # Load model and processor
        model, device = load_model()
        processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
        
        # Process the mosaic in patches
        mask = process_image_patches(mosaic, processor, model, device)
        
        # Save mask visualization
        mask_filename = f"roi_{int(time.time())}_mask.png"
        mask_path = save_mask_visualization(mask, mask_filename)
        
        # Create transform for the mosaic
        west, south, east, north = bounds
        transform = rasterio.transform.from_bounds(west, south, east, north, mosaic.width, mosaic.height)
        
        # Create a proper CRS for accurate area calculation
        crs = rasterio.crs.CRS.from_epsg(4326)
        
        # Convert mask to GeoJSON with proper coordinate system
        geojson_data = mask_to_geojson(mask, transform, crs)
        
        # Calculate solar potential using the same parameters as the main process
        geojson_with_solar = calculate_solar_potential(geojson_data)
        
        # Calculate totals
        total_area = sum(feature['properties']['area'] for feature in geojson_with_solar['features'])
        total_generation = geojson_with_solar['metadata']['total_annual_generation']
        total_power = geojson_with_solar['metadata']['total_instantaneous_power']
        
        # Format area for display
        formatted_area = f"{total_area:.1f} m²"
        if total_area > 10000:
            formatted_area = f"{total_area/10000:.2f} hectares ({total_area:.1f} m²)"
        
        logger.info(f"ROI Processing Results:")
        logger.info(f"Total Area: {formatted_area}")
        logger.info(f"Total Power: {total_power:.2f} W")
        logger.info(f"Total Generation: {total_generation:.2f} kWh/year")
        
        return jsonify({
            'success': True,
            'mask_file': mask_filename,
            'input_file': input_filename,
            'total_area': formatted_area,
            'total_area_m2': total_area,
            'total_generation': float(total_generation),
            'total_power': format_power(total_power),
            'geojson': geojson_with_solar
        })

    except Exception as e:
        logger.error(f"Error in process_roi: {str(e)}")
        logger.error(f"Stack trace: {sys.exc_info()}")
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def get_result(filename):
    """Serve result files"""
    return send_file(os.path.join(RESULTS_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True)
