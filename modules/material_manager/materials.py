import shutil
from modules.utils.constants import data_folders, project_folder


class Material:
    def __init__(self, name, file_path):
        self.name = name
        self.path = file_path

    def __str__(self):
        return self.name

    def add_material_from_file(self):
        material_folder = data_folders / self.name
        material_folder.mkdir(exist_ok=True, parents=True, mode=0o755)
        shutil.copy(str(self.path), str(material_folder))


class ProjectMaterials:
    def __init__(self):
        self.material_list = []
        self.find_materials()

    def find_materials(self):
        p = data_folders.glob('**/*')
        files = [x for x in p if x.is_file() and (".csv" in str(x))]
        for file in files:
            self.material_list.append(Material(file.parts[-2], file.relative_to(project_folder)))

    def get_material(self, material_name):
        if material_name == "":
            return Material("", "")
        for material in self.material_list:
            if material_name.lower() == material.name.lower():
                return material


if __name__ == '__main__':
    materials = ProjectMaterials()
    print(materials.material_list[0])
