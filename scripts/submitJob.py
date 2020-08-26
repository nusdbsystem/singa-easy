

from singa_auto import Client

# client = Client(admin_host='panda.d2.comp.nus.edu.sg', admin_port=3000)
client = Client(admin_host='ncrs.d2.comp.nus.edu.sg', admin_port=3000)
# client = Client(admin_host='localhost', admin_port=3000)
client.login(email='superadmin@singaauto', password='singa_auto')
print(client._user)


print(client.create_model(
    name='PyPandaVggV100',
    task='IMAGE_CLASSIFICATION',
    model_file_path='/Users//PyProjects/singa-easy/examples/models/image_classification/PyPandaVgg.py',
    model_class='PyPandaVgg',
    dependencies={"singa-easy": "0.4.1",
                  }
))

#
# print(client.create_dataset(
#                       name='xraytrain',
#                       task="IMAGE_CLASSIFICATION",
#                       dataset_path="/Users//Downloads/data/train.zip")
# )

# print(client.create_dataset(
#                       name='xrayval',
#                       task="IMAGE_CLASSIFICATION",
#                       dataset_path="/Users//Downloads/data/val.zip")
# )


print(client.create_train_job(app='PyPandaVggV100',
                              task='IMAGE_CLASSIFICATION',
                              train_dataset_id="c1f06925-01b9-4a19-a1bd-6120c5ef01a5",
                              val_dataset_id="c1f06925-01b9-4a19-a1bd-6120c5ef01a5",
                              models=['e2b01b55-4ecc-4d9a-981a-aca605a02385'],
                              budget={'MODEL_TRIAL_COUNT': 1},
                              train_args={}
                              ))
