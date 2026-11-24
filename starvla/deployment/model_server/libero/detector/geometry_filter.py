"""Category-free plane/fragment filtering. No ROI or semantic features.
For a pinhole camera, a plane satisfies 1/z = a*u+b*v+c even without K.
This is plane-consistency in predicted depth, NOT calibrated metric distance.
"""
import numpy as np
from scipy import ndimage as ndi
CONFIG=dict(relative_depth_tolerance=.025,ransac_trials=300,max_planes=4,min_plane_fraction=.12,min_background_fraction=.08,background_inlier_fraction=.92,ring_plane_fraction=.65,min_pixels=48,min_perforated_surface_fraction=.15,min_enclosed_holes=3,min_hole_pixels=24)
def plane_candidates(depth,seed=0):
 h,w=depth.shape;y,x=np.mgrid[:h,:w];A=np.stack([x/max(w-1,1),y/max(h-1,1),np.ones_like(x)],-1).reshape(-1,3)
 z=depth.reshape(-1);valid=np.isfinite(z)&(z>1e-4);remaining=valid.copy();rng=np.random.default_rng(seed);planes=[]
 for k in range(CONFIG['max_planes']):
  ids=np.flatnonzero(remaining)
  if len(ids)<h*w*CONFIG['min_plane_fraction']:break
  ids=rng.choice(ids,min(4000,len(ids)),replace=False);aa=A[ids];zz=z[ids];best=None;count=0
  for _ in range(CONFIG['ransac_trials']):
   pick=rng.choice(len(ids),3,replace=False)
   if abs(np.linalg.det(aa[pick]))<.003:continue
   coef=np.linalg.solve(aa[pick],1/zz[pick]);inv=aa@coef
   good=(inv>0)&(np.abs(inv*zz-1)<CONFIG['relative_depth_tolerance'])
   if good.sum()>count:count=int(good.sum());best=(coef,good)
  if best is None:break
  coef,good=best
  for _ in range(3):
   coef=np.linalg.lstsq(aa[good],1/zz[good],rcond=None)[0];good=np.abs((aa@coef)*zz-1)<CONFIG['relative_depth_tolerance']
   if good.sum()<3:break
  inv=A@coef;inlier=valid&(inv>0)&(np.abs(inv*z-1)<CONFIG['relative_depth_tolerance'])
  components,n=ndi.label((inlier&remaining).reshape(h,w));sizes=np.bincount(components.ravel());sizes[0]=0
  if not n or sizes.max()<h*w*CONFIG['min_plane_fraction']:break
  support=components==sizes.argmax()
  planes.append(dict(coef=coef.tolist(),support=support,inlier=inlier.reshape(h,w),fraction=float(support.mean())))
  remaining[support.ravel()]=False
 return planes

def classify_masks(depth,masks,quality):
 planes=plane_candidates(depth);size=depth.size;rows=[];areas=masks.sum((1,2));valid=np.isfinite(depth)&(depth>1e-4)
 for i,m in enumerate(masks):
  area=int(areas[i]);status='keep';reason='no_removal_evidence';evidence={}
  if not area:status='delete';reason='empty'
  elif (valid&m).sum()/area<.95:status='uncertain';reason='invalid_depth'
  elif area<CONFIG['min_pixels']:status='delete';reason='tiny_fragment'
  else:
   thickness=float(ndi.distance_transform_edt(m).max()*2)
   if area<160 and thickness<=2:status='delete';reason='thin_fragment'
   else:
    ring=ndi.binary_dilation(m,iterations=4)&~m
    holes=ndi.binary_fill_holes(m)&~m
    hole_ids,n_holes=ndi.label(holes);hole_sizes=np.bincount(hole_ids.ravel())[1:]
    substantial_holes=int((hole_sizes>=CONFIG['min_hole_pixels']).sum())
    for pidx,p in enumerate(planes):
     fraction=float(p['inlier'][m].mean());ring_frac=float(p['inlier'][ring].mean()) if ring.any() else 0
     outside=int((p['support']&~m).sum())/size
     if fraction<.8:continue
     evidence=dict(plane=pidx,plane_inlier_fraction=fraction,ring_plane_fraction=ring_frac,plane_support_outside_mask=outside,enclosed_holes=substantial_holes)
     if area/size>=CONFIG['min_background_fraction']:
      if fraction>=CONFIG['background_inlier_fraction'] and ring_frac>=CONFIG['ring_plane_fraction'] and outside>=.05:
       status='delete';reason='broad_continuous_plane';break
      if fraction>=CONFIG['background_inlier_fraction'] and area/size>=CONFIG['min_perforated_surface_fraction'] and substantial_holes>=CONFIG['min_enclosed_holes']:
       status='delete';reason='perforated_support_plane';break
      status='uncertain';reason='large_planar_surface'
     else:
      status='uncertain';reason='small_planar_region_preserved'
  rows.append(dict(region=i+1,status=status,reason=reason,area=area,**evidence))
 # Near-contained proposals only. Do not collapse clearly distinct nested items.
 for i,m in enumerate(masks):
  if rows[i]['status']=='delete':continue
  for j,p in enumerate(masks):
   if i==j or areas[j]<=areas[i] or rows[j]['status']=='delete':continue
   ratio=float(areas[i]/areas[j]);contain=float((m&p).sum()/max(areas[i],1))
   if contain<.95 or ratio<.65:continue
   parent_extra=p&~m
   dz=abs(float(np.median(depth[m]))-float(np.median(depth[parent_extra])))/max(float(np.median(depth[m])),1e-4)
   if dz<.02 and quality[j]>=quality[i]-.02:
    # A nested mask is not necessarily redundant; retain ambiguous parts.
    if ratio>=.8:rows[i].update(status='delete',reason='near_duplicate_containment',parent=j+1,area_ratio=ratio)
    else:rows[i].update(status='uncertain',reason='nested_part_preserved',parent=j+1,area_ratio=ratio)
    break
 return rows,planes
