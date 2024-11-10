import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# تولید داده‌های نمونه (به عنوان جایگزین برای داده‌های واقعی)
np.random.seed(42)
data_size = 200

# ورودی‌ها: مساحت، تعداد طبقات، ضریب عایق، تعداد پنجره‌ها، میانگین دما، سرعت باد، رطوبت نسبی
X = np.random.rand(data_size, 7) * [300, 3, 20, 10, 30, 20, 100]
y = X[:, 0] * 0.5 + X[:, 1] * 1.5 + X[:, 2] * 0.7 + X[:, 4] * 0.8 + np.random.randn(data_size) * 5

# تقسیم داده‌ها به مجموعه‌های آموزشی و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ساخت مدل شبکه عصبی
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# کامپایل مدل
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# آموزش مدل
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, validation_data=(X_test_scaled, y_test), verbose=0)

# ارزیابی مدل
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)

# پیش‌بینی با استفاده از داده‌های تست
predictions = model.predict(X_test_scaled)

print("Mean Absolute Error:", mae)
print("Sample Predictions:", predictions[:5])
