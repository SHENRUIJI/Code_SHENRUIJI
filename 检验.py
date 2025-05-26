import pkg_resources

def check_installed_packages(requirements_file):
    with open(requirements_file, 'r') as file:
        required_packages = [line.strip().split('==')[0] for line in file if not line.startswith('#')]

    installed_packages = {pkg.key for pkg in pkg_resources.working_set}

    for package in required_packages:
        if package.lower() in installed_packages:
            print(f"{package} 已安装")
        else:
            print(f"{package} 未安装")

if __name__ == "__main__":
    requirements_file = 'requirements.txt'
    check_installed_packages(requirements_file)
