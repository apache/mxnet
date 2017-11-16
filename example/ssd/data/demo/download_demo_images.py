import os

wd = os.path.dirname(os.path.realpath(__file__))

def download(url, target):
    os.system("wget {} -O {}".format(url, target))

if __name__ == "__main__":
    base_url = "https://cloud.githubusercontent.com/assets/3307514/"
    demo_list = {"20012566/cbb53c76-a27d-11e6-9aaa-91939c9a1cd5.jpg":"000001.jpg",
    "20012564/cbb43894-a27d-11e6-9619-ba792b66c4ae.jpg": "000002.jpg",
    "20012565/cbb53942-a27d-11e6-996c-125bb060a81d.jpg": "000004.jpg",
    "20012562/cbb4136e-a27d-11e6-884c-ed83c165b422.jpg": "000010.jpg",
    "20012567/cbb60336-a27d-11e6-93ff-cbc3f09f5c9e.jpg": "dog.jpg",
    "20012563/cbb41382-a27d-11e6-92a9-18dab4fd1ad3.jpg": "person.jpg",
    "20012568/cbc2d6f6-a27d-11e6-94c3-d35a9cb47609.jpg": "street.jpg"}
    for k, v in demo_list.items():
        download(base_url + k, os.path.join(wd, v))
