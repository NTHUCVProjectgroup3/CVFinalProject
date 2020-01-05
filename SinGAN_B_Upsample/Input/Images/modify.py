from PIL import Image
import os

filename = 'aachen'
a = Image.open('%s.png' % filename)
a_seg = Image.open('%s_seg.png' % filename)

out = a.resize((1024, 1024))
out_seg = a_seg.resize((1024, 1024), Image.NEAREST)

out.save('%s_out.png' % filename)
out_seg.save('%s_outseg.png' % filename)

a.close()
a_seg.close()

os.remove('%s.png' % filename)
os.remove('%s_seg.png' % filename)

os.rename('%s_out.png' % filename, '%s.png' % filename)
os.rename('%s_outseg.png' % filename, '%s_seg.png' % filename)