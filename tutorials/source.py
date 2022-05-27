import numpy as np

def update_header_from_cutout2D(hdu, cutout):
    # update data
    newdata = np.zeros((1,1,cutout.data.shape[0], cutout.data.shape[1]), dtype=np.float32)
    newdata[0,0,:,:] = cutout.data
    hdu.data = newdata
    # update header cards returned from cutout2D wcs:
    hdu.header.set('CRVAL1', cutout.wcs.wcs.crval[0])
    hdu.header.set('CRVAL2', cutout.wcs.wcs.crval[1])
    hdu.header.set('CRPIX1', cutout.wcs.wcs.crpix[0])
    hdu.header.set('CRPIX2', cutout.wcs.wcs.crpix[1])
    hdu.header.set('CDELT1', cutout.wcs.wcs.cdelt[0])
    hdu.header.set('CDELT2', cutout.wcs.wcs.cdelt[1])
    hdu.header.set('NAXIS1', cutout.wcs.pixel_shape[0])
    hdu.header.set('NAXIS2', cutout.wcs.pixel_shape[1])
    return hdu