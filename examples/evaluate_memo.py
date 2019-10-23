                    if net_iter == str(0):
                        #print(sample)
                        
                        #flow_uv = np.transpose(data['flow_gt'],(1,2,0))
                        flow_uv = data['flow_gt']
                        #print(flow_uv.shape)
                        np.set_printoptions(threshold=np.inf)
                        #print(flow_uv)
                        flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
                        print(sample)
                                              
                        plt.tick_params(labelbottom=False,
                                        labelleft=False,
                                        labelright=False,
                                        labeltop=False,
                                        bottom=False,
                                        left=False,
                                        right=False,
                                        top=False)
                        
                        plt.imshow(flow_color)
                        plt.savefig('/root/src/demon/examples/flow/' + str(sample) + '.png')
                        #print('saved' + str(sample) + '.jpg')
                        #plt.show()
                        
                        plt.clf()
                        plt.tick_params(labelbottom=False,
                                        labelleft=False,
                                        labelright=False,
                                        labeltop=False,
                                        bottom=False,
                                        left=False,
                                        right=False,
                                        top=False)
                        #plt.show()  
                        plt.imshow(data['depth_gt'],cmap='Greys')
                        plt.savefig('/root/src/demon/examples/depth/' + str(sample) + '.png')
                        plt.clf()
