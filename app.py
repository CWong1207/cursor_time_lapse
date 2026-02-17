from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import zipfile
import json
import base64
from io import BytesIO
from PIL import Image
from skimage.io import imread
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_closing, disk, remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from streamlit_segmentation_class import segmentation

app = Flask(__name__)
CORS(app)

# Store session data in memory (in production, use Redis or database)
sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and extract frames"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Create session
        session_id = request.form.get('session_id', 'default')
        if session_id not in sessions:
            sessions[session_id] = {
                'temp_dir': tempfile.mkdtemp(),
                'step': 'upload'
            }
        
        session = sessions[session_id]
        
        # Save uploaded file
        file_path = os.path.join(session['temp_dir'], file.filename)
        file.save(file_path)
        
        # Extract frames
        tiff = Image.open(file_path)
        output_dir = os.path.join(session['temp_dir'], 'all_pages')
        os.makedirs(output_dir, exist_ok=True)
        
        pages = []
        total_frames = tiff.n_frames
        
        for i in range(total_frames):
            tiff.seek(i)
            save_path = os.path.join(output_dir, f'page_{i + 1}.tiff')
            tiff.save(save_path)
            pages.append(save_path)
        
        session['pages'] = pages
        session['step'] = 'configure'
        
        return jsonify({
            'success': True,
            'total_frames': total_frames,
            'session_id': session_id
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/configure', methods=['POST'])
def configure():
    """Handle configuration (blocks, segment_index)"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        blocks = data.get('blocks', 5)
        segment_index = data.get('segment_index', 2)
        
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = sessions[session_id]
        pages = session['pages']
        
        # Organize into groups
        groups = {f'group{i + 1}': [] for i in range(blocks)}
        for i in range(len(pages)):
            block_num = (i % blocks) + 1
            groups[f'group{block_num}'].append(pages[i])
        
        session['groups'] = groups
        session['blocks'] = blocks
        session['segment_index'] = segment_index
        session['step'] = 'sigma'
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preview_sigma', methods=['POST'])
def preview_sigma():
    """Generate sigma preview images"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = sessions[session_id]
        segment_index = session.get('segment_index', 2)
        pages = session['pages']
        
        nuclei_frame = imread(pages[segment_index])
        segment_variable = segmentation(nuclei_frame)
        
        # Generate previews for sigmas [3, 4, 5]
        sigmas = [3, 4, 5]
        preview_images = []
        
        for sigma in sigmas:
            blurred = gaussian(nuclei_frame, sigma=sigma)
            thresh = threshold_otsu(blurred.flatten())
            nuclei_mask = blurred > thresh
            nuclei_mask = binary_closing(nuclei_mask, disk(1))
            nuclei_mask = remove_small_objects(nuclei_mask, min_size=500)
            nuclei_mask = binary_fill_holes(nuclei_mask)
            labeled_nuclei = label(nuclei_mask)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(nuclei_frame, cmap="Reds", alpha=0.6)
            ax.imshow(nuclei_mask, cmap="nipy_spectral", alpha=0.4)
            ax.set_title(f"Sigma={sigma}\n({len(np.unique(labeled_nuclei)) - 1} nuclei)")
            ax.axis("off")
            
            # Add labels
            props = regionprops(labeled_nuclei)
            for prop in props:
                y, x = prop.centroid
                ax.text(x, y, str(prop.label), color="white", fontsize=8,
                       ha="center", va="center",
                       bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=0.6))
            
            # Convert to base64
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            preview_images.append({
                'sigma': sigma,
                'image': f'data:image/png;base64,{img_base64}',
                'nuclei_count': len(np.unique(labeled_nuclei)) - 1
            })
            plt.close(fig)
        
        return jsonify({'success': True, 'previews': preview_images})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/confirm_sigma', methods=['POST'])
def confirm_sigma():
    """Confirm sigma selection and perform segmentation"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        sigma = data.get('sigma', 3)
        
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = sessions[session_id]
        segment_index = session.get('segment_index', 2)
        pages = session['pages']
        
        nuclei_frame = imread(pages[segment_index])
        segment_variable = segmentation(nuclei_frame)
        nuclei_mask = segment_variable.nucleus_segment(sigma=sigma)
        
        # Store nuclei_mask as numpy array (convert to list for JSON)
        session['nuclei_mask'] = nuclei_mask.tolist()
        session['nuclei_frame_shape'] = list(nuclei_frame.shape)
        session['step'] = 'dilation'
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preview_dilation', methods=['POST'])
def preview_dilation():
    """Generate dilation preview - compute once and cache"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = sessions[session_id]
        
        # Check if already computed
        if 'dilation_rings' not in session:
            segment_index = session.get('segment_index', 2)
            pages = session['pages']
            nuclei_frame = imread(pages[segment_index])
            
            # Reconstruct nuclei_mask from stored list
            nuclei_mask = np.array(session['nuclei_mask'])
            
            segment_variable = segmentation(nuclei_frame)
            segment_variable.nuclei_mask = nuclei_mask
            
            # Compute rings for all dilation factors
            dilation_factors = [4, 8, 12, 16]
            rings_dict = segment_variable.compute_dilation_rings(dilation_factors)
            
            # Convert to base64 images and store
            preview_images = []
            for d in dilation_factors:
                all_rings = rings_dict[d]
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(nuclei_frame, cmap="gray")
                ax.imshow(nuclei_mask > 0, cmap="Blues", alpha=0.4)
                ax.imshow(all_rings, cmap="Oranges", alpha=0.5)
                ax.set_title(f"Dilation={d}")
                ax.axis("off")
                
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                preview_images.append({
                    'dilation': d,
                    'image': f'data:image/png;base64,{img_base64}'
                })
                plt.close(fig)
            
            session['dilation_rings'] = preview_images
        
        return jsonify({
            'success': True,
            'previews': session['dilation_rings']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/confirm_dilation', methods=['POST'])
def confirm_dilation():
    """Confirm dilation selection"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        dilation = data.get('dilation', 8)
        
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = sessions[session_id]
        session['dilation'] = dilation
        session['step'] = 'ready'
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/run_analysis', methods=['POST'])
def run_analysis():
    """Run the analysis on selected layer"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        layer = data.get('layer', 1)
        
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = sessions[session_id]
        groups = session['groups']
        dilation = session['dilation']
        
        # Reconstruct nuclei_mask
        nuclei_mask = np.array(session['nuclei_mask'])
        
        images_group = [imread(p) for p in groups[f'group{layer}']]
        results = []
        
        for i, image in enumerate(images_group):
            segment_variable = segmentation([image])
            segment_variable.nuclei_mask = nuclei_mask
            df = segment_variable.protein_quantification(dilation_factor=dilation)
            df['Frame'] = i + 1
            results.append(df)
        
        # Combine results
        final_df = pd.concat(results, ignore_index=True)
        final_df['Group'] = layer
        final_df = final_df.sort_values(by=['Group', 'NucleusLabel', 'Frame'])
        
        # Generate plots
        plots_folder = os.path.join(session['temp_dir'], 'plots')
        os.makedirs(plots_folder, exist_ok=True)
        plot_files = []
        
        unique_labels = np.sort(final_df['NucleusLabel'].unique())
        for label in unique_labels:
            data = final_df[final_df['NucleusLabel'] == label]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(data['NucleusMeanRFP'], label='Nucleus', marker='o')
            ax.plot(data['CytoplasmMeanRFP'], label='Cytoplasm', marker='s')
            ax.set_title(f"Nucleus {label}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Mean Intensity")
            ax.legend()
            ax.grid(True, alpha=0.3)
            p_file = os.path.join(plots_folder, f'{label}.png')
            plt.savefig(p_file, dpi=150, bbox_inches='tight')
            plot_files.append(p_file)
            plt.close()
        
        # Save CSV
        csv_path = os.path.join(session['temp_dir'], 'data.csv')
        final_df.to_csv(csv_path, index=False)
        
        # Create ZIP
        zip_path = os.path.join(session['temp_dir'], 'results.zip')
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(csv_path, 'measurements.csv')
            for pf in plot_files:
                zf.write(pf, os.path.basename(pf))
        
        # Return results as JSON and ZIP path
        return jsonify({
            'success': True,
            'results': final_df.to_dict('records'),
            'zip_path': zip_path
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<session_id>')
def download_file(session_id):
    """Download the results ZIP file"""
    try:
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session = sessions[session_id]
        zip_path = os.path.join(session['temp_dir'], 'results.zip')
        
        if not os.path.exists(zip_path):
            return jsonify({'error': 'Results file not found'}), 404
        
        return send_file(zip_path, as_attachment=True, download_name='results.zip')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Configuration - prioritize environment variables (set by cloud platforms)
    # Cloud platforms (Railway, Render, Heroku) set PORT automatically
    PORT = int(os.getenv('PORT', os.getenv('FLASK_PORT', 5000)))
    HOST = os.getenv('HOST', os.getenv('FLASK_HOST', '0.0.0.0'))  # 0.0.0.0 for cloud platforms
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'  # Default False for production
    
    # Try to load from config.py for local development
    try:
        from config import HOST as CFG_HOST, PORT as CFG_PORT, DEBUG as CFG_DEBUG
        # Only use config if environment variables not set
        if 'PORT' not in os.environ and 'FLASK_PORT' not in os.environ:
            PORT = CFG_PORT
        if 'HOST' not in os.environ and 'FLASK_HOST' not in os.environ:
            HOST = CFG_HOST
        if 'FLASK_DEBUG' not in os.environ:
            DEBUG = CFG_DEBUG
    except ImportError:
        pass  # Use defaults above
    
    print("=" * 50)
    print("Flask Server Starting...")
    print("=" * 50)
    print(f"Local access: http://localhost:{PORT}")
    
    if HOST == '0.0.0.0':
        import socket
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
            print(f"Network access: http://{local_ip}:{PORT}")
            print(f"Hostname access: http://{hostname}:{PORT}")
        except:
            pass
    
    # Check for custom domain in config
    try:
        from config import CUSTOM_DOMAIN
        if CUSTOM_DOMAIN:
            print(f"Custom domain: http://{CUSTOM_DOMAIN}:{PORT}")
    except:
        pass
    
    print("=" * 50)
    print(f"Press Ctrl+C to stop the server")
    print("=" * 50)
    print()
    
    app.run(host=HOST, port=PORT, debug=DEBUG)
