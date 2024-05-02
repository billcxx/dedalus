```
Created on Tue Jun 23 23:27:04 2015
@author: cxx
@var filename

```
import os
from vtk import *
filename="strain.vtk"
this dir=os.path.dirname(os.path.abspath( file ))vtk path=os.path.join(this dir,filename)
reader =vtkDataSetReader()
reader.SetFileName(vtk path)reader.Update()
mapper =vtkSmartVolumeMapper()
mapper.SetInputconnection(reader.GetOutputPort())
actor = vtkVolume()
actor.SetMapper(mapper)
renderer =vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(1,1,1)
renderer window =vtkRenderwindow()renderer window.AddRenderer(renderer)