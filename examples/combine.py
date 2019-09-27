import matplotlib

for i in range(300):
   rgb = mpimg.imread(str(i) + '.png')
   gt = mpimg.imread(str(i) + '.png')
   depth = mpimg.imread(str(i) + '.png')
 
   #plt.figure(1)
   plt.subplot(311)
   plt.imshow(rgb)
   
   plt.subplot(312)
   plt.imshow(gt)

   plt.subplot(313)
   plt.imshow(depth)
   plt.show()
   
   plt.savefig(str(i) + '.png')
