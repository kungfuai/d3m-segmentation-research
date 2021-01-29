import torch

from src.dataset.big_earth_dataset import BAND_STATS

def normalize(img_dict, band_key, size):
    x = img_dict[band_key].reshape((size, size))
    x = (x - BAND_STATS['mean'][band_key]) / BAND_STATS['std'][band_key]
    return torch.tensor(x, dtype=torch.float32)

def preprocess(
    img_dict,
    one_image_label=False,
    one_pixel_mask=False
):
    B01  = normalize(img_dict, 'B01', 21)
    B02  = normalize(img_dict, 'B02', 126)
    B03  = normalize(img_dict, 'B03', 126)
    B04  = normalize(img_dict, 'B04', 126)
    B05  = normalize(img_dict, 'B05', 63)
    B06  = normalize(img_dict, 'B06', 63)
    B07  = normalize(img_dict, 'B07', 63)
    B08  = normalize(img_dict, 'B08', 126)
    B8A  = normalize(img_dict, 'B8A', 63)
    B09  = normalize(img_dict, 'B09', 21)
    B11  = normalize(img_dict, 'B11', 63)
    B12  = normalize(img_dict, 'B12', 63)
    bands_10m = torch.stack([B04, B03, B02, B08], axis=2)
    bands_20m = torch.stack([B05, B06, B07, B8A, B11, B12], axis=2)
    bands_60m = torch.stack([B01, B09], axis=2)
    
    bands_20m = torch.unsqueeze(bands_20m.permute(2,0,1), 0)
    bands_20m = torch.nn.functional.interpolate(
        bands_20m, 
        size=(126,126),
        mode='bicubic',
        align_corners=False
    )
    bands_20m = torch.squeeze(bands_20m).permute(1,2,0)

    bands_60m = torch.unsqueeze(bands_60m.permute(2,0,1), 0)
    bands_60m = torch.nn.functional.interpolate(
        bands_60m, 
        size=(126,126),
        mode='bicubic',
        align_corners=False
    )
    bands_60m = torch.squeeze(bands_60m).permute(1,2,0)

    img = torch.cat(
        [bands_10m, bands_20m, bands_60m], 
        axis=2
    ).permute(2,0,1)

    img = torch.nn.functional.pad(
        img, 
        (1,1,1,1,0,0)
    )

    if one_image_label:
        labels = torch.tensor(img_dict['class_label'], dtype=torch.float32)
        return img, labels
    
    else:
        labels = img_dict['Corine_labels'].reshape((126, 126))
        labels = torch.tensor(labels, dtype=torch.float32)
        
        if one_pixel_mask:
            labels_one = torch.zeros((126,126), dtype=torch.float32)
            mask = torch.zeros((126,126), dtype=torch.float32)

            pi, pj = img_dict['random_pixel']
            labels_one[pi][pj] = labels[pi][pj]
            mask[pi][pj] = torch.ones((1,))
            labels = labels_one

        else:    
            mask = torch.ones((126,126), dtype=torch.int32)
                    
        labels = torch.nn.functional.pad(labels, (1,1,1,1))
        mask = torch.nn.functional.pad(mask, (1,1,1,1))

        labels = torch.unsqueeze(labels, 0)
        mask = torch.unsqueeze(mask, 0)

        return img, labels, mask