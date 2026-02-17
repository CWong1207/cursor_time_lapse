from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_closing, disk, remove_small_objects, dilation
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class segmentation:
    def __init__(self, images):
        self.images = images
        self.nuclei_mask = None

    def preview_segmentation_options(self, sigmas=[3, 4, 5]):
        """
        Original method to compare different Gaussian blur (sigma) levels.
        """
        results = {}
        fig, axes = plt.subplots(1, len(sigmas), figsize=(15, 5))

        if len(sigmas) == 1:
            axes = [axes]

        for idx, sigma in enumerate(sigmas):
            # Perform segmentation with this sigma
            blurred = gaussian(self.images, sigma=sigma)
            thresh = threshold_otsu(blurred.flatten())
            nuclei_mask = blurred > thresh
            nuclei_mask = binary_closing(nuclei_mask, disk(1))
            nuclei_mask = remove_small_objects(nuclei_mask, min_size=500)
            nuclei_mask = binary_fill_holes(nuclei_mask)
            labeled_nuclei = label(nuclei_mask)

            # Store result
            results[sigma] = labeled_nuclei

            # Plot
            axes[idx].imshow(self.images, cmap="Reds", alpha=0.6)
            axes[idx].imshow(nuclei_mask, cmap="nipy_spectral", alpha=0.4)
            axes[idx].set_title(f"Sigma={sigma}\n({len(np.unique(labeled_nuclei)) - 1} nuclei)")
            axes[idx].axis("off")

            # Add labels
            props = regionprops(labeled_nuclei)
            for prop in props:
                y, x = prop.centroid
                axes[idx].text(x, y, str(prop.label), color="white", fontsize=8,
                               ha="center", va="center",
                               bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=0.6))

        plt.tight_layout()
        # Return the dictionary of results so the UI can use them if needed
        return results

    def compute_dilation_rings(self, dilation_factors=[4, 8, 12]):
        """
        Compute cytoplasmic rings for different dilation factors.
        Returns a dictionary mapping dilation_factor -> ring_mask.
        This can be cached separately from plotting.
        """
        rings_dict = {}
        
        for d in dilation_factors:
            # Create a blank mask to accumulate all cytoplasmic rings for this dilation factor
            all_rings = np.zeros_like(self.nuclei_mask, dtype=bool)

            for region_label in np.unique(self.nuclei_mask):
                if region_label == 0: continue

                nucleus_mask = self.nuclei_mask == region_label
                # Create the ring using the user-specified factor 'd'
                outer_ring = dilation(nucleus_mask, disk(d))
                inner_ring = dilation(nucleus_mask, disk(2))
                cytoplasm_mask = outer_ring & ~inner_ring
                all_rings = all_rings | cytoplasm_mask
            
            rings_dict[d] = all_rings
        
        return rings_dict

    def preview_dilation_options(self, dilation_factors=[4, 8, 12], rings_dict=None):
        """
        NEW: Visualizes the cytoplasmic ring based on the mask created in the previous step.
        If rings_dict is provided, uses cached computation instead of recomputing.
        """
        fig, axes = plt.subplots(1, len(dilation_factors), figsize=(15, 5))
        if len(dilation_factors) == 1: axes = [axes]

        # Use the single frame for the preview overlay
        img = self.images if not isinstance(self.images, list) else self.images[0]

        # Use cached rings if provided, otherwise compute them
        if rings_dict is None:
            rings_dict = self.compute_dilation_rings(dilation_factors)

        for idx, d in enumerate(dilation_factors):
            all_rings = rings_dict[d]
            
            axes[idx].imshow(img, cmap="gray")
            # Overlay nuclei in Blue and Rings in Orange
            axes[idx].imshow(self.nuclei_mask > 0, cmap="Blues", alpha=0.4)
            axes[idx].imshow(all_rings, cmap="Oranges", alpha=0.5)
            axes[idx].set_title(f"Dilation={d}")
            axes[idx].axis("off")

        plt.tight_layout()
        return fig

    def nucleus_segment(self, sigma):
        """
        Perform final nucleus segmentation.
        Note: We removed the default =3 so the calling code MUST provide the user's choice.
        """
        blurred = gaussian(self.images, sigma=sigma)
        thresh = threshold_otsu(blurred.flatten())
        nuclei_mask = blurred > thresh
        nuclei_mask = binary_closing(nuclei_mask, disk(1))
        nuclei_mask = remove_small_objects(nuclei_mask, min_size=500)
        nuclei_mask = binary_fill_holes(nuclei_mask)
        labeled_nuclei = label(nuclei_mask)

        # Store the result in the class instance
        self.nuclei_mask = labeled_nuclei
        return labeled_nuclei

    def protein_quantification(self, dilation_factor):
        """
        Calculate intensities using the user-selected dilation_factor.
        Default =8 removed to ensure user selection is explicitly passed.
        """
        results = []
        imgs = self.images if isinstance(self.images, list) else [self.images]

        for i in range(len(imgs)):
            for region_label in np.unique(self.nuclei_mask):
                if region_label == 0:
                    # skip background
                    continue

                nucleus_mask = self.nuclei_mask == region_label

                # test to see whether nucleus_mask touches the border of the image
                ys, xs = np.where(nucleus_mask)
                touches_border = (
                        ys.min() == 0 or ys.max() == imgs[i].shape[0] - 1 or
                        xs.min() == 0 or xs.max() == imgs[i].shape[1] - 1
                )

                if touches_border:
                    results.append({
                        "NucleusLabel": region_label,
                        "NucleusMeanRFP": np.nan,
                        "CytoplasmMeanRFP": np.nan,
                        "NucleusMeanRFP/CytoplasmMeanRFP": np.nan,
                        "IncompleteMask": True
                    })
                    continue

                # define cytoplasmic ring using the factor passed from the UI
                outer_ring = dilation(nucleus_mask, disk(dilation_factor))
                inner_ring = dilation(nucleus_mask, disk(2))
                cytoplasm_mask = outer_ring & ~inner_ring

                # intensity measurements
                nucleus_intensity = imgs[i][nucleus_mask].mean()
                cytoplasm_intensity = imgs[i][cytoplasm_mask].mean()

                if cytoplasm_intensity == 0 or nucleus_intensity == 0:
                    ratio = np.nan
                else:
                    ratio = nucleus_intensity / cytoplasm_intensity

                results.append({
                    "NucleusLabel": region_label,
                    "NucleusMeanRFP": nucleus_intensity,
                    "CytoplasmMeanRFP": cytoplasm_intensity,
                    "NucleusMeanRFP/CytoplasmMeanRFP": ratio
                })

        return pd.DataFrame(results)
