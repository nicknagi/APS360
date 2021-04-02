demo_transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

test_data_dir = "/content/demo2/DEMO"
demo_set = datasets.ImageFolder(test_data_dir, transform=demo_transform) 

demo_loader = DataLoader(demo_set)
print(f"Number of demo examples: {len(demo_loader)}")

classes = ['Organic', 'Recyclable']
batch_size = 1
num_sample_batches = 2
num_cols = 2
num_rows = num_sample_batches / num_cols

demo_iter = iter(demo_loader)
fig = plt.figure(figsize=(25, 10))

for i in range(num_sample_batches):
    for j in range(batch_size):
      imgs, labels = demo_iter.next()
      ax = fig.add_subplot(num_rows, num_cols, i + 1)
      plt.imshow(np.transpose(imgs[j], (1, 2, 0)))
      ax.set_title(classes[labels[j]])

print(f"Demo Accuracy: {test_accuracy(model, demo_loader):.8f}")