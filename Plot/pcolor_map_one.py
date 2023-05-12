# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:18:56 2017

@author: Xiaohui
"""

import os
import numpy as np
#import cartopy.crs as ccrs
#import cartopy
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import font_manager
from  PIL import Image, ImageFont, ImageDraw
#from cartopy.feature import NaturalEarthFeature, LAND, COASTLINE
#import cartopy.mpl.ticker as cticker
#from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
#import cartoee as cee

def pcolor_map_one(X, Y, Var_plot, title_temp, Xlabel, Ylabel, pathName, options):
        
    if 'fontsize' not in options.keys():
        #Set default fontsize to be 25   
        options['fontsize'] = 20
        
    if 'linewidth' not in options.keys():
        #Set default linewidth to be 1
        options['linewidth'] = 1        
    
    #The numofVar specifies the number of variables/times to be plot
    numofVar = len(Var_plot) 
     
    if ('size_figure' in options.keys()) == 0 | ('position' in options.keys() == 0):
        #Set the size of figures, and position vector of the subplots based on 
        #the number of variables.
        if numofVar == 1:
            options['size_figure'] = [10, 8]
            if title_temp == []:
                options['position'] = [[0.1, 0.1, 0.7, 0.8], 
                                       [0.825, 0.1, 0.025, 0.8]]
               
            else:
                options['position'] = [[0.1, 0.1, 0.7, 0.7], 
                                       [0.825, 0.1, 0.025, 0.7]]
                
            options['subplot'] = [111]
            
            if ('proj' in options.keys()) & ('lon_0' in options.keys()):
                options['position'] = [[0.02, 0.02, 0.87, 0.87], 
                                       [0.90, 0.02, 0.02, 0.87]]
                
    
        elif numofVar == 2:
            options['size_figure'] = [18, 8]
            options['position'] = [[0.035, 0.05, 0.4, 0.8],
                                   [0.5, 0.05, 0.4, 0.8],
                                   [0.915,0.05,0.015,0.8]]
            options['subplot'] = [121, 122]
            
        elif numofVar == 3:
            '''
            options['size_figure'] = [12, 27]
            options['position'] = [[0.075, 0.033, 0.75, 0.28],
                                   [0.075, 0.363, 0.75, 0.28],
                                   [0.075, 0.693, 0.75, 0.28],
                                   [0.85, 0.033, 0.025, 0.94]]
            options['subplot'] = [311, 312, 313]
            '''
            options['size_figure'] = [30, 12]
            options['position'] = [[0.033, 0.075, 0.28, 0.75],
                                   [0.343, 0.075, 0.28, 0.75],
                                   [0.653, 0.075, 0.28, 0.75],
                                   [0.943, 0.075, 0.02, 0.75]]
            options['subplot'] = [311, 321, 331]
            
        elif numofVar == 4:
            options['size_figure'] = [20, 20]       
            options['position'] = [[0.033, 0.05, 0.4, 0.4],
                                   [0.033, 0.5, 0.4, 0.4],
                                   [0.5, 0.05, 0.4, 0.4],
                                   [0.5, 0.5, 0.4, 0.4],
                                   [0.943, 0.05, 0.02, 0.9]] 
            options['subplot'] = [221, 222, 223, 224]   

        elif (numofVar == 5) | (numofVar == 6):
            options['size_figure'] = [30, 20]         
               
            options['subplot'] = [321, 322, 323, 324, 325, 326]   
        
    else:
        options['size_figure'] = [20, 20]
        
            
    #Plot
    hf = plt.figure( figsize = (options['size_figure'][0], options['size_figure'][1]) )
    
    axes = []
    caxes = []
    
    #Loop through all the variables
    for i in range(numofVar):
        
        plt.rc('font', size = options['fontsize'])
        
        if 'quiver' in options.keys():
            if Var_plot[i]['U'].size == 0:
                continue;
                
            if 'skip' not in options.keys():
                options['skip'] = 1
        
        else:
            if Var_plot[i].size == 0 :
                continue;

        #Create subplots
        if 'proj' in options.keys():
            if 'position' in options.keys():
                cmd = "axes.append( hf.add_axes(options['position'][i], projection = ccrs." + options['proj'] + "()) )"
            else:
                cmd = "axes.append( hf.add_subplot(options['subplot'][i], projection = ccrs." + options['proj'] + "()) )"
             
            eval(cmd)
            
        else:            
            if 'position' in options.keys():
                axes.append( hf.add_axes(options['position'][i]) )
            else:
                axes.append( hf.add_subplot(options['subplot'][i]) )            
            
        plt.sca(axes[i])
                            
        if 'proj' in options.keys():
            if 'extent' in options.keys():
                axes[i].set_extent(options['extent'])
                
            ''            
            axes[i].coastlines(resolution = '50m')
            
            #Compute map of projected grid.
            print('#{} shape 1 {}, shape 2 {}'.format(i, X[i].shape, Var_plot[i].shape))
            
            if 'Colorbar_Tick' in options.keys():
                cmd = "axes[i].pcolor( X[i], Y[i], Var_plot[i], cmap = 'jet', \
                vmin = int(options['Colorbar_Tick'][0]), vmax = int(options['Colorbar_Tick'][-1]), \
                transform = ccrs." + options['proj'] + "()  )"      
          
            else:
                cmd = "axes[i].pcolor( X[i], Y[i], Var_plot[i], cmap = 'jet', \
                transform = ccrs." + options['proj'] + "()  )"
    
            eval(cmd)
            
            # draw coastlines, country boundaries 
            axes[i].coastlines(resolution = '50m', color='black', linewidth = 1)
            #axes[i].add_feature(cartopy.feature.COASTLINE)
            axes[i].add_feature(cartopy.feature.BORDERS, edgecolor='black')
            #axes[i].add_feature(cartopy.feature.OCEAN)            

            '''
            # data resolution
            resol = '50m'
            
            # province boundaries
            provinc_bodr = cartopy.feature.NaturalEarthFeature(category='cultural', 
            name='admin_1_states_provinces_lines', scale=resol, facecolor='none', edgecolor='k')
            
            axes[i].add_feature(provinc_bodr, linestyle='--', linewidth=0.6, edgecolor="k", zorder=10)
            '''
            axes[i].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
            axes[i].yaxis.set_major_formatter(LATITUDE_FORMATTER)
            
            # Define the xticks for longitude
            cmd = "axes[i].set_xticks(np.arange(np.ceil(X[i].min()), X[i].max(), 2).astype(int), crs=ccrs." + options['proj'] + "() )"
            #axes[i].set_xticks(np.arange(114, 124, 2), crs=ccrs.Mercator())
 #           cmd = "axes[i].set_xticks(np.arange(np.ceil(X[i].min()), X[i].max(), 2).astype(int), crs=ccrs." + options['proj'] + "() )"
            eval(cmd)
            
            # Define the yticks for latitude
            cmd = "axes[i].set_yticks(np.arange(np.ceil(Y[i].min()), Y[i].max(), 2).astype(int), crs=ccrs." + options['proj'] + "() )"            
            eval(cmd)            
            
            ''
            if ('Colorbar' in options.keys()):
                if options['Colorbar'] == True:

                    cax = axes[i].figure.add_axes(options['position'][-1])
                    
                    visualization = {}
                    
                    if 'Colorbar_label' in options.keys():
                        visualization['bands'] = options['Colorbar_label']
                            
                    if 'Colorbar_Tick' in options.keys():
                        visualization['min'] = int(options['Colorbar_Tick'][0])
                        visualization['max'] = int(options['Colorbar_Tick'][-1])
                                             
                    #cb = cee.addColorbar(axes[i], loc='right', cmap = plt.get_cmap('jet'), visParams = visualization)
                    cb = cee.addColorbar(axes[i], cax = cax, cmap = plt.get_cmap('jet'), visParams = visualization)
                        
            ''
        elif 'quiver' in options.keys():                

            caxes.append( plt.contourf(X[i], Y[i], Var_plot[i]['WS'], cmap = 'jet') )
                
            if 'Colorbar_Tick' in options.keys():
                plt.clim(options['Colorbar_Tick'][0], options['Colorbar_Tick'][-1])
            
            plt.quiver(X[i][::options['skip'], ::options['skip']], Y[i][::options['skip'], ::options['skip']], \
            Var_plot[i]['U'][::options['skip'], ::options['skip']], Var_plot[i]['V'][::options['skip'], ::options['skip']], \
            pivot='middle') 
            
        else:
                                    
            if 'alpha' in options.keys():
                caxes.append( plt.pcolor(X[i], Y[i], Var_plot[i], alpha = options['alpha'], cmap = 'jet') ) 
            else:
                caxes.append( plt.pcolor(X[i], Y[i], Var_plot[i], cmap = 'jet') ) 
                
        if 'Colormap' in options.keys():
            plt.set_cmap(options['Colormap'])   
                
        #plt.show()
        
        plt.rc('font', size = options['fontsize'])
            
        #Set title, x, and y labels
        if len(title_temp) != 0:
            axes[i].set_title(title_temp[i], fontsize = options['fontsize'])
            #.title(title_temp[i], fontsize = options['fontsize'])

        if Xlabel != '':
            plt.xlabel(Xlabel, fontsize = options['fontsize'])
        
        if Ylabel != '':
            plt.ylabel(Ylabel, fontsize = options['fontsize'])

        #Check if the following options are specified or not  
        if 'contour' in options.keys():
            plt.contour(X[i], Y[i], options['contour'][i], np.arange(1, options['contour'][i].max()+1))
            
        if 'Colormap_Contour' in options.keys():
            plt.set_cmap(options['Colormap_Contour'])                
        
        if ('lat_station' in options.keys()) & ('lon_station' in options.keys()):
            ListofColors = ['ws', 'ws', 'wo', 'w+']
            #plt.plot(options['lon_station'], options['lat_station'], 'w+')            
            #Plot locations of stations            
            for n in range(len(options['lat_station'])):
                print('plot stations')
                print(len(options['lon_station'][n]))
                                      
        '''
        if 'Colorbar_Tick' in options.keys():
            plt.clim(options['Colorbar_Tick'][0], options['Colorbar_Tick'][-1])
            
        if 'XTick' in options.keys():
            plt.xticks(options['XTick'])
        if 'Xlim' in options.keys():
            plt.xlim(options['Xlim'][0], options['Xlim'][1])
            
        if 'YTick' in options.keys():
            plt.yticks(options['YTick'])
        if 'Ylim' in options.keys():
            plt.ylim(options['Ylim'][0], options['Ylim'][1])
        '''
        if ('lat_boundary' in options.keys()) & ('lon_boundary' in options.keys()):
            plt.plot(options['lon_boundary'], options['lat_boundary'], 'k-', linewidth = options['linewidth'])                

        if ('title_temp_chinese' in options.keys()) & ('fontPath' in options.keys()):
            #This is for fixing Chinese words display in the figure
            fontP = font_manager.FontProperties(fname = options['fontPath'])
            fontP.set_size(options['fontsize'])    
            
            plt.title(options['title_temp_chinese'], fontproperties = fontP)

    #Activate colorbar and set label to be Ylabel_Var
    #print( "length of options['position']: " + str(len(options['position'])) )
    '''
    if ('Colorbar' in options.keys()):
        if options['Colorbar'] == True:
            
            if 'Colorbar_label' in options.keys():
                if ('position' in options.keys() != 0):                                    
                    cbar = plt.colorbar( caxes[i], cax = hf.add_axes(options['position'][numofVar]), \
                    label = options['Colorbar_label'] )
           
                else:
                    cbar = plt.colorbar( caxes[i], label = options['Colorbar_label'] )
                    
            else:          
                #Have colorbar by default.
                if ('position' in options.keys() != 0):                                                    
                    cbar = hf.colorbar( caxes[i] , cax = hf.add_axes(options['position'][numofVar]) )
 
                else:
                    cbar = hf.colorbar( caxes[i] )
                    
            if 'Colorbar_Tick' in options.keys():
                cbar.set_ticks(options['Colorbar_Tick'])
                    
            if 'Colorbar_title' in options.keys():
                cbar.ax.set_title(options['Colorbar_title'])
    '''
        
    #Save figure
    if 'transparent' not in options.keys():
        options['transparent'] = False
            
    if 'text' in options.keys():         
        for n in range(len(options['text']['x'])):
            if np.mod(n, 2) == 0:
                plt.text(options['text']['x'][n], options['text']['y'][n], options['text']['text'][n], \
                         horizontalalignment='center', verticalalignment='top', transform = axes[-1].transData, fontsize=6)                                
        
            else:
                plt.text(options['text']['x'][n], options['text']['y'][n], options['text']['text'][n], \
                         horizontalalignment='center', verticalalignment='bottom', transform = axes[-1].transData, fontsize=6)                                
                
    if 'dpi' in options.keys():
        hf.savefig(pathName, dpi = options['dpi'], transparent = options['transparent'])
    else:
        hf.savefig(pathName, transparent = options['transparent'])        
    
    '''
    #Add text on the figure.
    if 'text' in options.keys(): 
        pwd = os.getcwd()
        #font_temp = ImageFont.truetype(os.path.join(pwd, '..', 'FY2', 'times.ttf'), 80)                
        
        image = Image.open(pathName)
        draw = ImageDraw.Draw(image)                            
        #draw.text((options['text']['x'], options['text']['y']), options['text']['text'], (0,0,0), font = fontP)                        
#        draw.text((options['text']['x'], options['text']['y']), options['text']['text'], options['text']['color'], font = font_temp)                        
        for n in range(len(options['text']['x'])):
            draw.text((options['text']['x'][n], options['text']['y'][n]), options['text']['text'][n], transform = axes[-1].transData)                                
        image.save(pathName,  transparent = True)
    '''
    
    print ("Figure %s is saved." % pathName)
    plt.clf()
    plt.cla()
    plt.close(hf) 

def pcolor_map_one_python(X, Y, Var_plot, title_temp, Xlabel, Ylabel, pathName, options):
        
    if 'fontsize' not in options.keys():
        #Set default fontsize to be 25   
        options['fontsize'] = 20
        
    if 'linewidth' not in options.keys():
        #Set default linewidth to be 1
        options['linewidth'] = 1        
    
    #The numofVar specifies the number of variables/times to be plot
    numofVar = len(Var_plot) 
        
    if ('size_figure' in options.keys()) == 0 | ('position' in options.keys() == 0):
        #Set the size of figures, and position vector of the subplots based on 
        #the number of variables.
        if numofVar == 1:
            options['size_figure'] = [10, 8]
            if title_temp == []:
                options['position'] = [[0.1, 0.1, 0.7, 0.8], 
                                       [0.825, 0.1, 0.025, 0.8]]
               
            else:
                options['position'] = [[0.1, 0.1, 0.7, 0.7], 
                                       [0.825, 0.1, 0.025, 0.7]]
                
            options['subplot'] = [111]
            
            if ('proj' in options.keys()) & ('lon_0' in options.keys()):
                options['position'] = [[0.02, 0.02, 0.87, 0.87], 
                                       [0.90, 0.02, 0.02, 0.87]]
                
    
        elif numofVar == 2:
            options['size_figure'] = [18, 8]
            options['position'] = [[0.035, 0.05, 0.4, 0.8],
                                   [0.5, 0.05, 0.4, 0.8],
                                   [0.915,0.05,0.015,0.8]]
            options['subplot'] = [121, 122]
            
        elif numofVar == 3:
            '''
            options['size_figure'] = [12, 27]
            options['position'] = [[0.075, 0.033, 0.75, 0.28],
                                   [0.075, 0.363, 0.75, 0.28],
                                   [0.075, 0.693, 0.75, 0.28],
                                   [0.85, 0.033, 0.025, 0.94]]
            options['subplot'] = [311, 312, 313]
            '''
            options['size_figure'] = [30, 12]
            options['position'] = [[0.033, 0.075, 0.28, 0.75],
                                   [0.343, 0.075, 0.28, 0.75],
                                   [0.653, 0.075, 0.28, 0.75],
                                   [0.943, 0.075, 0.02, 0.75]]
            options['subplot'] = [311, 321, 331]
            
        elif numofVar == 4:
            options['size_figure'] = [20, 20]       
            options['position'] = [[0.05, 0.5, 0.4, 0.4],
                                   [0.5, 0.5, 0.4, 0.4],
                                   [0.05, 0.05, 0.4, 0.4],
                                   [0.5, 0.05, 0.4, 0.4],
                                   [0.943, 0.05, 0.02, 0.9]]
            options['subplot'] = [221, 222, 223, 224]   

        elif (numofVar == 5) | (numofVar == 6):
            options['size_figure'] = [30, 20]         
               
            options['subplot'] = [321, 322, 323, 324, 325, 326]   
        
    else:
        options['size_figure'] = [20, 20]
        
            
    #Plot
    hf = plt.figure( figsize = (options['size_figure'][0], options['size_figure'][1]) )
    
    axes = []
    caxes = []
    
    #Loop through all the variables
    for i in range(numofVar):
        
        plt.rc('font', size = options['fontsize'])
        
        if 'quiver' in options.keys():
            if Var_plot[i]['U'].size == 0:
                continue;
                
            if 'skip' not in options.keys():
                options['skip'] = 1
        
        else:
            if Var_plot[i].size == 0 :
                continue;

        #Create subplots
        if 'proj' in options.keys():
            if 'position' in options.keys():
                axes.append(hf.add_axes(options['position'][i], projection = getattr(ccrs, options['proj'])()))
            else:
                axes.append(hf.add_subplot(options['subplot'][i], projection = getattr(ccrs, options['proj'])()))            
        else:            
            if 'position' in options.keys():
                axes.append( hf.add_axes(options['position'][i]) )
            else:
                axes.append( hf.add_subplot(options['subplot'][i]) )            
            
        plt.sca(axes[i])
                            
        if 'proj' in options.keys():
            if 'extent' in options.keys():
                axes[i].set_extent(options['extent'])
                
            axes[i].coastlines(resolution = '50m')
            
            #Compute map of projected grid.
            print('#{} shape 1 {}, shape 2 {}'.format(i, X[i].shape, Var_plot[i].shape))
            
            if 'Colorbar_Tick' in options.keys():
                axes[i].pcolor( X[i], Y[i], Var_plot[i], cmap = 'jet', \
                    vmin = int(options['Colorbar_Tick'][0]), vmax = int(options['Colorbar_Tick'][-1]), \
                    transform = getattr(ccrs, options['proj'])())
            else:
                axes[i].pcolor(X[i], Y[i], Var_plot[i], cmap = 'jet', \
                    transform = getattr(ccrs, options['proj'])())
            
            # draw coastlines, country boundaries 
            axes[i].coastlines(resolution = '50m', color='black', linewidth = 1)
            #axes[i].add_feature(cartopy.feature.COASTLINE)
            axes[i].add_feature(cartopy.feature.BORDERS, edgecolor='black')
            #axes[i].add_feature(cartopy.feature.OCEAN)            

            '''
            # data resolution
            resol = '50m'
            
            # province boundaries
            provinc_bodr = cartopy.feature.NaturalEarthFeature(category='cultural', 
            name='admin_1_states_provinces_lines', scale=resol, facecolor='none', edgecolor='k')
            
            axes[i].add_feature(provinc_bodr, linestyle='--', linewidth=0.6, edgecolor="k", zorder=10)
            '''
            axes[i].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
            axes[i].yaxis.set_major_formatter(LATITUDE_FORMATTER)
            
            # Define the xticks for longitude
            axes[i].set_xticks(np.arange(np.ceil(X[i].min()), X[i].max(), 2).astype(int), crs=getattr(ccrs, options['proj'])() )
            
            # Define the yticks for latitude
            axes[i].set_yticks(np.arange(np.ceil(Y[i].min()), Y[i].max(), 2).astype(int), crs=getattr(ccrs, options['proj'])() )
            
            ''
            if ('Colorbar' in options.keys()):
                if options['Colorbar'] == True:

                    cax = axes[i].figure.add_axes(options['position'][-1])
                    
                    visualization = {}
                    
                    if 'Colorbar_label' in options.keys():
                        visualization['bands'] = options['Colorbar_label']
                            
                    if 'Colorbar_Tick' in options.keys():
                        visualization['min'] = int(options['Colorbar_Tick'][0])
                        visualization['max'] = int(options['Colorbar_Tick'][-1])
                                             
                    #cb = cee.addColorbar(axes[i], loc='right', cmap = plt.get_cmap('jet'), visParams = visualization)
                    cb = cee.addColorbar(axes[i], cax = cax, cmap = plt.get_cmap('jet'), visParams = visualization)
                        
            ''
        elif 'quiver' in options.keys():                

            caxes.append( plt.contourf(X[i], Y[i], Var_plot[i]['WS'], cmap = 'jet') )
                
            if 'Colorbar_Tick' in options.keys():
                plt.clim(options['Colorbar_Tick'][0], options['Colorbar_Tick'][-1])
            
            plt.quiver(X[i][::options['skip'], ::options['skip']], Y[i][::options['skip'], ::options['skip']], \
            Var_plot[i]['U'][::options['skip'], ::options['skip']], Var_plot[i]['V'][::options['skip'], ::options['skip']], \
            pivot='middle') 
            
        else:
                                    
            if 'alpha' in options.keys():
                caxes.append( plt.pcolor(X[i], Y[i], Var_plot[i], alpha = options['alpha'], cmap = 'jet') ) 
            else:
                caxes.append( plt.pcolor(X[i], Y[i], Var_plot[i], cmap = 'jet') ) 
                
        if 'Colormap' in options.keys():
            plt.set_cmap(options['Colormap'])   
                
        #plt.show()
        
        plt.rc('font', size = options['fontsize'])
            
        #Set title, x, and y labels
        if len(title_temp) != 0:
            axes[i].set_title(title_temp[i], fontsize = options['fontsize'])
            #.title(title_temp[i], fontsize = options['fontsize'])

        if Xlabel != '':
            plt.xlabel(Xlabel, fontsize = options['fontsize'])
        
        if Ylabel != '':
            plt.ylabel(Ylabel, fontsize = options['fontsize'])

        #Check if the following options are specified or not  
        if 'contour' in options.keys():
            plt.contour(X[i], Y[i], options['contour'][i], np.arange(1, options['contour'][i].max()+1))
            
        if 'Colormap_Contour' in options.keys():
            plt.set_cmap(options['Colormap_Contour'])                
        
        if ('lat_station' in options.keys()) & ('lon_station' in options.keys()):
            ListofColors = ['ws', 'ws', 'wo', 'w+']
            #plt.plot(options['lon_station'], options['lat_station'], 'w+')            
            #Plot locations of stations            
            for n in range(len(options['lat_station'])):
                print('plot stations')
                print(len(options['lon_station'][n]))
                                      
        '''
        if 'Colorbar_Tick' in options.keys():
            plt.clim(options['Colorbar_Tick'][0], options['Colorbar_Tick'][-1])
            
        if 'XTick' in options.keys():
            plt.xticks(options['XTick'])
        if 'Xlim' in options.keys():
            plt.xlim(options['Xlim'][0], options['Xlim'][1])
            
        if 'YTick' in options.keys():
            plt.yticks(options['YTick'])
        if 'Ylim' in options.keys():
            plt.ylim(options['Ylim'][0], options['Ylim'][1])
        '''
        if ('lat_boundary' in options.keys()) & ('lon_boundary' in options.keys()):
            plt.plot(options['lon_boundary'], options['lat_boundary'], 'k-', linewidth = options['linewidth'])                

        if ('title_temp_chinese' in options.keys()) & ('fontPath' in options.keys()):
            #This is for fixing Chinese words display in the figure
            fontP = font_manager.FontProperties(fname = options['fontPath'])
            fontP.set_size(options['fontsize'])    
            
            plt.title(options['title_temp_chinese'], fontproperties = fontP)

    #Activate colorbar and set label to be Ylabel_Var
    #print( "length of options['position']: " + str(len(options['position'])) )
    '''
    if ('Colorbar' in options.keys()):
        if options['Colorbar'] == True:
            
            if 'Colorbar_label' in options.keys():
                if ('position' in options.keys() != 0):                                    
                    cbar = plt.colorbar( caxes[i], cax = hf.add_axes(options['position'][numofVar]), \
                    label = options['Colorbar_label'] )
           
                else:
                    cbar = plt.colorbar( caxes[i], label = options['Colorbar_label'] )
                    
            else:          
                #Have colorbar by default.
                if ('position' in options.keys() != 0):                                                    
                    cbar = hf.colorbar( caxes[i] , cax = hf.add_axes(options['position'][numofVar]) )
 
                else:
                    cbar = hf.colorbar( caxes[i] )
                    
            if 'Colorbar_Tick' in options.keys():
                cbar.set_ticks(options['Colorbar_Tick'])
                    
            if 'Colorbar_title' in options.keys():
                cbar.ax.set_title(options['Colorbar_title'])
    '''
        
    #Save figure
    if 'transparent' not in options.keys():
        options['transparent'] = False
            
    if 'text' in options.keys():         
        for n in range(len(options['text']['x'])):
            if np.mod(n, 2) == 0:
                plt.text(options['text']['x'][n], options['text']['y'][n], options['text']['text'][n], \
                         horizontalalignment='center', verticalalignment='top', transform = axes[-1].transData, fontsize=6)                                
        
            else:
                plt.text(options['text']['x'][n], options['text']['y'][n], options['text']['text'][n], \
                         horizontalalignment='center', verticalalignment='bottom', transform = axes[-1].transData, fontsize=6)                                
                
    if 'dpi' in options.keys():
        hf.savefig(pathName, dpi = options['dpi'], transparent = options['transparent'])
    else:
        hf.savefig(pathName, transparent = options['transparent'])        
    
    '''
    #Add text on the figure.
    if 'text' in options.keys(): 
        pwd = os.getcwd()
        #font_temp = ImageFont.truetype(os.path.join(pwd, '..', 'FY2', 'times.ttf'), 80)                
        
        image = Image.open(pathName)
        draw = ImageDraw.Draw(image)                            
        #draw.text((options['text']['x'], options['text']['y']), options['text']['text'], (0,0,0), font = fontP)                        
#        draw.text((options['text']['x'], options['text']['y']), options['text']['text'], options['text']['color'], font = font_temp)                        
        for n in range(len(options['text']['x'])):
            draw.text((options['text']['x'][n], options['text']['y'][n]), options['text']['text'][n], transform = axes[-1].transData)                                
        image.save(pathName,  transparent = True)
    '''
    
    print ("Figure %s is saved." % pathName)
    plt.clf()
    plt.cla()
    plt.close(hf) 
