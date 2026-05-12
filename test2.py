from test1 import testmethod1

print(testmethod1())


from pathlib import Path

database_path = Path("mtgDir")

for file_path in database_path.glob('*'):
    print(file_path.stem)