#Level 0 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv")
X = df.drop('label', axis=1).values  # This drops the label column and gives only the pixel data 784 columns
y = df['label'].values  # This gives only the label column which has values from 0-9
print(X.shape)
print(y.shape)
unique_labels = np.unique(y) #Gives all unique values from label column in an array
print("Unique labels:", unique_labels)  
label_counts = pd.Series(y).value_counts() #Converting numpy array into pandas series and checking the frequency of each label i.e clothing category
print("Label distribution:\n", label_counts)
# Selecting the first image and reshaping it to 28x28
sample_image = X[0].reshape(28, 28)
# Displaying the image
plt.imshow(sample_image, cmap='gray')
plt.title(f"Label: {y[0]}")
plt.show()
print("Pixel value range:", np.min(sample_image), "to", np.max(sample_image))
#Level 1
for i in range(10):  # Loop through labels 0-9
    # Find an image with label i
    sample_idx = np.where(y == i)[0][0] #Taking the 0th index image just to get an idea of which category it belongs to
    sample_image = X[sample_idx].reshape(28, 28)
    plt.subplot(2, 5, i+1)
    plt.imshow(sample_image, cmap='gray')
    plt.title(f"Label {i}")
    plt.axis('off')
plt.show()

pixel_stats = pd.DataFrame(X).describe() #Converting numpy array into pandas dataframe to use describe
print("Summary statistics for pixel values:\n", pixel_stats)
plt.hist(X.flatten(), bins=50, color='gray')
plt.title("Distribution of Pixel Values")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

#Level 2
from sklearn.preprocessing import StandardScaler
# Normalize pixel values
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
# Verifying normalization by seeing the min and max range
print("Normalized pixel value range:", np.min(X_normalized), "to", np.max(X_normalized))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
from sklearn.linear_model import LogisticRegression
# Initializing and training the model
model = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=0)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report
# Predict on test set
y_pred = model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
import shap
# Create a SHAP explainer (this requires additional setup and may be complex for large datasets)
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)
# Visualize SHAP values for a sample
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test[0,:], matplotlib=True)