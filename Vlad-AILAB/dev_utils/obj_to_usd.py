from pxr import Usd, UsdGeom, Gf, Sdf
import os


def convert_obj_to_usd(obj_file_path, usd_file_path):
    # Create a new USD stage
    stage = Usd.Stage.CreateNew(usd_file_path)

    # Define the path for the mesh in the USD file
    mesh_path = Sdf.Path("/Mesh")

    # Create a mesh prim at the specified path
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)

    # Read the .obj file
    with open(obj_file_path, "r") as obj_file:
        lines = obj_file.readlines()

    vertices = []
    faces = []

    for line in lines:
        if line.startswith("v "):
            # Vertex position
            parts = line.strip().split()
            vertices.append(Gf.Vec3f(float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            # Face indices (assuming triangular faces)
            parts = line.strip().split()
            face_indices = [
                int(part.split("/")[0]) - 1 for part in parts[1:]
            ]  # Convert to 0-based index
            faces.append(face_indices)

    # Set the vertices and faces on the mesh
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([idx for face in faces for idx in face])

    # Save the USD stage to disk
    stage.GetRootLayer().Save()

    print(f"Converted {obj_file_path} to {usd_file_path}")


# Given a folder path, convert all .obj files in the folder to USD files with the same name but in a different path
def convert_all_objs_to_usd(folder_path, output_folder):
    obj_files = [f for f in os.listdir(folder_path) if f.endswith(".obj")]
    for obj_file in obj_files:
        obj_file_path = os.path.join(folder_path, obj_file)
        usd_file_path = os.path.join(output_folder, obj_file.replace(".obj", ".usd"))
        convert_obj_to_usd(obj_file_path, usd_file_path)


# Example usage
# convert_obj_to_usd('path/to/input.obj', 'path/to/output.usd')
convert_all_objs_to_usd("objects/blocks", "objects/blocks_usd")
