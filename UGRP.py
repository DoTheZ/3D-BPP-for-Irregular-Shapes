import os
import numpy as np
import trimesh
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

    
class Space:
    def __init__(self, box_size, resolution):
        self.width = box_size[0]
        self.height = box_size[1]
        self.depth = box_size[2]
        self.resolution = resolution
        self.empty = True
        
        self.shape = [int(box_size[i] / self.resolution) for i in range(3)]
        self.grid = np.zeros([self.shape[0], self.shape[1], self.shape[2]], dtype = int)

        self.heightMap = np.zeros([self.shape[0], self.shape[1]], dtype = int)

        self.point_indices = []
        
        self.packed_list = []
        self.volume = box_size[0] * box_size[1] * box_size[2]
        self.packed_volume = 0
    
    def pack(self, obj):

        self.check_possible_rotations(obj)
        print("0. Checked Possible Rotations")
    
        self.getHeightMap()
        print("1. Got HeightMap")

        valid_rot_and_pos = [] # will contain valid positions for each rotated objects
        
        count = 0
        for obj_rot in obj.rotations:
            valid_pos = self.get_valid_pos(obj_rot)
            valid_rot_and_pos.append(valid_pos)
            count += 1
            print("Done: {} / {}".format(count, len(obj.rotations)))
        print("2. Checked Valid positions")
        
        if(not all(sublist == [] for sublist in valid_rot_and_pos)):
            optimal_rot, optimal_pos = self.find_optimal(valid_rot_and_pos, obj)
            print("3. Found Optimal!")
            
            self.put_obj(optimal_rot, optimal_pos, obj, mode = 'optimal')

            print("========Packed : {}========".format(obj.name))

            self.packed_volume += obj.volume

            print("Current Packed Ratio: {}".format(self.packed_volume / self.volume))
        
        else:
            print("XXXXXXXXXXXXXXXX Failed : {} XXXXXXXXXXXXXXxXXX".format(obj.name))
            print("{} / {}".format(obj.shape[0], self.shape[0]))
            print("{} / {}".format(obj.shape[1], self.shape[1]))
            print("{} / {}".format(obj.shape[2], self.shape[2]))
  
    def check_possible_rotations(self, obj):
        possible_rotations = []
        for obj_rot in obj.rotations:
            if obj_rot.shape[0] > self.shape[0] or obj_rot.shape[1] > self.shape[1] or obj_rot.shape[2] > self.shape[2]:
                continue
            possible_rotations.append(obj_rot)
        obj.rotations = possible_rotations
  
    def getHeightMap(self):
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in reversed(range(self.shape[2])):
                    if (self.grid[x][y][z] == 0):
                        continue
                    elif(self.grid[x][y][z] > 0):
                        self.heightMap[x][y] = z
                        break

    def get_valid_pos(self, obj_rot):
        if self.empty:
            return [(0,0,0)]
        
        min_idx = np.argmin(self.heightMap)
        min_idx = np.unravel_index(min_idx, self.heightMap.shape)
        min_height = self.heightMap[min_idx]

        obj_bottom_size = [obj_rot.shape[0], obj_rot.shape[1]]
        obj_bottom = np.zeros(obj_bottom_size, dtype = int)
        obj_bottom = obj_rot[:,:,0]

        valid_pos = []

        if (min_height > (self.shape[2] - obj_rot.shape[2])):
            return valid_pos
        
        for z in range(int(min_height), (self.shape[2] - obj_rot.shape[2] + 1)):
            xy_plane = self.grid[:,:, z]
            for y in range(0,(self.shape[1] - obj_rot.shape[1] + 1)):
                for x in range(0,(self.shape[0] - obj_rot.shape[0] + 1)):

                    # 1. Check Whether the object can be put on the xy-plane
                    slice_x = slice(x, x + obj_rot.shape[0])
                    slice_y = slice(y, y + obj_rot.shape[1])
                    xy_plane[slice_x, slice_y] += obj_bottom
                    
                    if np.any(xy_plane[slice_x, slice_y] > obj.index):
                        xy_plane[slice_x, slice_y] -= obj_bottom
                        continue

                    xy_plane[slice_x, slice_y] -= obj_bottom

                    # 2. Check whether the object can be positioned
                    slice_z = slice(z, z + obj_rot.shape[2])
                    self.grid[slice_x, slice_y, slice_z] += obj_rot

                    if np.any(self.grid[slice_x, slice_y, slice_z] > obj.index):
                        self.grid[slice_x, slice_y, slice_z] -= obj_rot
                        continue
                    self.grid[slice_x, slice_y, slice_z] -= obj_rot

                    valid_pos.append((x,y,z))
            if len(valid_pos) > 0:
                break
        return valid_pos
        
    def find_optimal(self, rot_and_pos, obj):
        area = [[] for _ in range(len(rot_and_pos))]
        rot_num = 0
        for rot in rot_and_pos:
            if len(rot):
                for pos in rot:

                    obj_grid = [(pos[i], pos[i] + obj.rotations[rot_num].shape[i])  for i in range(3)]

                    area[rot_num].append(self.getContactedArea(rot_num, pos, obj))

            else:
                area[rot_num].append(-1)

            rot_num += 1
            print("Done: {} / {}".format(rot_num, len(rot_and_pos)))

        local_max_area = []
        max_idx_list = []
        for rot in area:
            max_idx = np.argmax(rot)
            local_max_area.append(rot[max_idx])
            max_idx_list.append(max_idx)

        optimal_rot = np.argmax(local_max_area)
        optimal_pos = rot_and_pos[optimal_rot][max_idx_list[optimal_rot]]
     
        return optimal_rot, optimal_pos
    
    def getRotatedSurface(self, rot_num, pos, obj):
        surface = obj.default_surface
        
        rot_i = rot_num % 16
        rot_j = (rot_num - rot_i) % 4
        rot_k = (rot_num - rot_i - 4*rot_j) % 4
        rotated = np.rot90(surface, k=rot_i, axes=(0, 1)) 
        rotated = np.rot90(rotated, k=rot_j, axes=(1, 2)) 
        rotated = np.rot90(rotated, k=rot_k, axes=(0, 2)) 

        surface_rotated = []
        for x in range(rotated.shape[0]):
            for y in range(rotated.shape[1]):
                for z in range(rotated.shape[2]):
                    if(rotated[x,y,z] == obj.index):
                        surface_rotated.append((x+pos[0], y+pos[1], z+pos[2]))
        return surface_rotated

    def getContactedArea(self, rot_num, pos, obj): 

        surface = self.getRotatedSurface(rot_num, pos, obj)   
        area = 0
        border_list = [[0,self.shape[0]-1],
                           [0,self.shape[1]-1],
                           [0]]
        for point in surface:

            if ((point[0] in border_list[0]) or (point[1] in border_list[1]) or (point[2] in border_list)):
                area += 1
                continue

            cube_range = [slice(point[i]-1, point[i]+1) for i in range(3)]
            point_cube = self.grid[tuple(cube_range)]

            if np.any(np.logical_and(point_cube > 0, point_cube != obj.index)):
                area += 1
                continue
        
        return area
    
    def put_obj(self, rot, pos, obj, mode = 'normal'):
        obj_grid = obj.rotations[rot]

        slice_x = slice(pos[0], pos[0] + obj_grid.shape[0])
        slice_y = slice(pos[1], pos[1] + obj_grid.shape[1])
        slice_z = slice(pos[2], pos[2] + obj_grid.shape[2])
        
        self.grid[slice_x, slice_y, slice_z] += obj_grid
        
        if mode == 'optimal':
            obj_points = []
            ranges = [range(pos[i], pos[i] + obj_grid.shape[i]) for i in range(3)]

            for z in ranges[2]:
                for y in ranges[1]:
                    for x in ranges[0]:
                        if self.grid[x][y][z] == obj.index:
                            pos = (x,y,z)
                            obj_points.append(pos)
                self.point_indices.append(obj_points)
            self.empty = False
            self.packed_list.append(obj.name)    

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for indices in self.point_indices:
            indices_len = len(indices)
            x = [indices[i][0] for i in range(indices_len)]
            y = [indices[i][1] for i in range(indices_len)]
            z = [indices[i][2] for i in range(indices_len)]
            rgb = [np.random.randint(0,255) / 255 for _ in range(3)]
            color = [tuple(rgb) for _ in range(indices_len)]
            ax.scatter(x,y,z,c=color,marker='o')

        ax.set_xlim(0, self.shape[0])
        ax.set_ylim(0, self.shape[1])
        ax.set_zlim(0, self.shape[2])
        ax.set_box_aspect([self.shape[0], self.shape[1], self.shape[2]])

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        plt.show()


class Object:
    def __init__(self, file_path, index, resolution):
        self.index = index
        self.resolution = resolution

        self.name = os.path.basename(file_path)
        self.mesh = trimesh.load_mesh(file_path)
        self.volume = self.mesh.volume
        self.mesh = self.mesh.voxelized(pitch = self.resolution)

        self.shape = [self.mesh.shape[i] for i in range(3)]
        print("name: {}".format(self.name))
        print("shape: {}".format(self.shape))
        self.grid = np.zeros(self.shape, dtype=int)

        for voxel in self.mesh.sparse_indices:
            self.grid[tuple(voxel)] = index
        
        self.default_surface = np.zeros(self.shape, dtype=int)
        self.inside= np.ones(self.shape, dtype = int)
        self.getSurface()
        self.fill_inside()
        
        self.rotations = []
        self.rotate()

    def getSurface(self):
        surface= []
        ranges = [range(0, self.shape[i]) for i in range(3)]
                       
        # xy-plane 1
        for x in ranges[0]:
            for y in ranges[1]:
                for z in ranges[2]:
                    self.inside[x,y,z] = 0
                    if self.grid[x,y,z] == self.index:
                        surface.append((x,y,z))
                        break
        # yz-plane 1
        for z in ranges[2]:
            for y in ranges[1]:
                for x in ranges[0]:
                    self.inside[x,y,z] = 0
                    if self.grid[x,y,z] == self.index:
                        surface.append((x,y,z))
                        break
        # xz-plane 1
        for z in ranges[2]:
            for x in ranges[0]:
                for y in ranges[1]:
                    self.inside[x,y,z] = 0
                    if self.grid[x,y,z] == self.index:
                        surface.append((x,y,z))
                        break
        # xy-plane 2
        for x in reversed(ranges[0]):
            for y in reversed(ranges[1]):
                for z in reversed(ranges[2]):
                    self.inside[x,y,z] = 0
                    if self.grid[x,y,z] == self.index:
                        surface.append((x,y,z))
                        break
        
        # yz-plane 2
        for z in reversed(ranges[2]):
            for y in reversed(ranges[1]):
                for x in reversed(ranges[0]):
                    self.inside[x,y,z] = 0
                    if self.grid[x,y,z] == self.index:
                        surface.append((x,y,z))
                        break

        # xz-plane 2
        for x in reversed(ranges[0]):
            for z in reversed(ranges[2]):
                for y in reversed(ranges[1]):
                    self.inside[x,y,z] = 0
                    if self.grid[x,y,z] == self.index:
                        surface.append((x,y,z))
                        break

        surface = list(set(surface)) # 중복요소 제거)

        for x in ranges[0]:
            for y in ranges[1]:
                for z in ranges[2]:
                    pos = (x,y,z)
                    if pos in surface:
                        self.default_surface[pos] = self.index

        
    
    def fill_inside(self):
        inside_points = []

        ranges = [range(0, self.shape[i]) for i in range(3)]
        for x in ranges[0]:
            for y in ranges[1]:
                for z in ranges[2]:
                    if self.grid[x,y,z] == 0 and self.inside[x,y,z] == self.index:
                        pos = (x,y,z)
                        inside_points.append(pos)

        percent = 0.2
        inside_points = random.sample(inside_points, int(percent * len(inside_points)))
        for pos in inside_points:
            self.grid[pos] = self.index

    # x, y, z축으로 각각 90도씩 회전
    def rotate(self):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    grid_rotated = np.rot90(self.grid, k=i, axes=(0, 1)) 
                    grid_rotated = np.rot90(grid_rotated, k=j, axes=(1, 2)) 
                    grid_rotated = np.rot90(grid_rotated, k=k, axes=(0, 2)) 

                    self.rotations.append(grid_rotated)
    
def load_objects(source_path, resolution):
    obj_list = []
    total = 0
    for file in os.listdir(source_path):
        if(file[-4:] == ".obj"):
            total += 1
    print("# of objects: {}".format(total))
    
    count = 0
    for file in os.listdir(source_path):
        if(file[-4:] == ".obj"):
            file_path = os.path.join(source_path, file)
            count += 1
            obj_list.append(Object(file_path, count, resolution))
            print("--------------- Processed {} / {} ------------------".format(count, total))

    return obj_list

def sort_objects(objects, criterion = "volume"):
    value_list = []

    # 1. sort by volume
    if criterion == "volume":
        for obj in objects:
            value_list.append(obj.volume)
    # 2. sort by bounding box size
    elif criterion == "bounding_box":
        for obj in objects:
            value_list.append(obj.shape[0] * obj.shape[1] * obj.shape[2])
    else:
        raise ValueError("Not Expected Criterion")

    sorted_objects = []
    length = len(value_list)
    for _ in range(length):
        max_idx = np.argmax(value_list)
        value_list.pop(max_idx)
        sorted_objects.append(objects[max_idx])
        objects.pop(max_idx)
    
    for obj in sorted_objects:
        if criterion == "bounding_box":
            print("{}, size : {}".format(obj.name, obj.shape[0] * obj.shape[1] * obj.shape[2]))
        elif criterion == "volume":
            print("{}, size : {}".format(obj.name, obj.volume))

    return sorted_objects


# Main Process starts from Here #

start_time = time.time()

source_path = "./objects_revised/processed" # Change Path
objects = []

resolution = 0.005 # You can change the resolution, but 0.005 is recommended
objects = load_objects(source_path, resolution)

box_size = (0.33, 0.37, 0.30) # You can chane box size
box = Space(box_size, resolution)

objects = sort_objects(objects, "bounding_box") # You can change "volume" or "bounding_box"

print("Start Packing...")


start_time = time.time()
for obj in objects:
    start_one = time.time()
    box.pack(obj)
    finish_one = time.time()
    packing_one = finish_one - start_one
    print("Packing One: {}".format(packing_one))
finish_time = time.time()
packing_time = finish_time - start_time
print("Packing Time: {}".format(packing_time))
print("Packed Ratio: {}".format(box.packed_volume / box.volume))
print("Packed List: {}".format(box.packed_list))
box.visualize()
