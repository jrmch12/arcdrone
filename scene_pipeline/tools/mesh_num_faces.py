import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mesh_reduction.mesh_utils import list_nb_faces

if __name__ == "__main__":
    ## List number of faces
    list_nb_faces("assets/hand3fingers/assets/merged")