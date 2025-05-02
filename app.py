import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import io
import base64

st.set_page_config(layout="wide")
st.title("🔍 Streamlit Data Analytics Tool")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🧹 Data Cleaning")
    st.write("Missing Values:", df.isnull().sum())
    clean_option = st.radio("Pilih metode pembersihan:", ("Drop rows", "Fill with mean", "Do nothing"))
    if clean_option == "Drop rows":
        df = df.dropna()
    elif clean_option == "Fill with mean":
        df = df.fillna(df.mean(numeric_only=True))
    st.dataframe(df.head())

    st.subheader("📊 Descriptive Analytics")
    st.write(df.describe())
    st.markdown("""
    **Cara Baca:**
    - **count**: Jumlah nilai yang tersedia
    - **mean**: Rata-rata
    - **std**: Standar deviasi (penyebaran data)
    - **min/max**: Nilai minimum dan maksimum
    - **25%/50%/75%**: Kuartil (pembagian data menjadi 4 bagian)
    """)

    st.subheader("📊 Visualisasi Data Interaktif")
    st.markdown("Pilih jenis chart dan kolom yang ingin divisualisasikan:")

    chart_type = st.selectbox(
        "Pilih jenis chart:",
        ("Bar Chart", "Line Chart", "Pie Chart")
    )

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    if chart_type in ["Bar Chart", "Line Chart"]:
        chart_cols = st.multiselect(
            "Pilih kolom numerik untuk visualisasi:",
            num_cols,
            default=num_cols[:2] if len(num_cols) >= 2 else num_cols
        )
    elif chart_type == "Pie Chart":
        chart_cols = st.multiselect(
            "Pilih satu kolom (kategorikal/ numerik):",
            df.columns.tolist(),
            default=[cat_cols[0]] if cat_cols else [num_cols[0]]
        )

    # Visualisasi
    if chart_type == "Bar Chart":
        if len(chart_cols) >= 1:
            st.bar_chart(df[chart_cols])
        else:
            st.info("Pilih minimal satu kolom numerik untuk bar chart.")

    elif chart_type == "Line Chart":
        if len(chart_cols) >= 1:
            st.line_chart(df[chart_cols])
        else:
            st.info("Pilih minimal satu kolom numerik untuk line chart.")

    elif chart_type == "Pie Chart":
        if len(chart_cols) == 1:
            col = chart_cols[0]
            fig, ax = plt.subplots()
            if df[col].dtype == 'object':
                counts = df[col].value_counts()
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
            else:
                # Untuk numerik, kelompokkan jadi kategori
                counts = pd.cut(df[col], bins=5).value_counts()
                ax.pie(counts, labels=counts.index.astype(str), autopct='%1.1f%%')
            ax.set_title(f"Pie Chart: {col}")
            st.pyplot(fig)
        else:
            st.info("Pilih satu kolom saja untuk pie chart.")

    st.subheader("🧪 Diagnostic Analytics")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    selected_cols = st.multiselect("Pilih kolom untuk korelasi:", num_cols, default=num_cols)
    if selected_cols:
        fig, ax = plt.subplots()
        sns.heatmap(df[selected_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        st.markdown("""
        **Cara Baca Heatmap Korelasi:**
        - Nilai mendekati **1** berarti korelasi positif kuat
        - Nilai mendekati **-1** berarti korelasi negatif kuat
        - Nilai mendekati **0** berarti tidak ada korelasi
        """)

    st.subheader("📈 Predictive Analytics (Regression)")
    target_col = st.selectbox("🎯 Pilih kolom target (numerik):", num_cols)
    feature_cols = st.multiselect("🧩 Pilih kolom fitur:", [col for col in num_cols if col != target_col])
    if target_col and feature_cols:
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("MSE:", mean_squared_error(y_test, y_pred))
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_test, y_pred)
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        st.pyplot(fig2)
        st.markdown("""
        **Cara Baca Grafik:**
        - Titik-titik mendekati garis diagonal berarti prediksi mendekati nilai aktual
        - Semakin menyebar, semakin besar error model
        """)

    st.subheader("🧭 Prescriptive Analytics")
    if feature_cols:
        input_data = {col: st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean())) for col in feature_cols}
        input_df = pd.DataFrame([input_data])
        predicted = model.predict(input_df)[0]
        st.success(f"Prediksi {target_col}: {predicted:.2f}")
        st.markdown("""
        **Penjelasan:**
        Anda bisa mengubah nilai fitur dan melihat hasil prediksi target secara langsung. Cocok untuk simulasi atau keputusan berbasis data.
        """)

    st.subheader("🎯 Classification Model")
    cat_target = st.selectbox("Target klasifikasi (kategori):", df.select_dtypes(include='object').columns.tolist())
    cat_features = st.multiselect("Fitur numerik untuk klasifikasi:", num_cols)
    if cat_target and cat_features:
        X = df[cat_features]
        y = df[cat_target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        st.write("Akurasi:", accuracy_score(y_test, y_pred))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        fig3, ax3 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax3)
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")
        st.pyplot(fig3)
        st.markdown("""
        **Cara Baca Confusion Matrix:**
        - Baris: label aktual, Kolom: prediksi
        - Nilai tinggi di diagonal = prediksi akurat
        - Nilai di luar diagonal = kesalahan klasifikasi
        """)

    st.subheader("🔍 Clustering Analysis")
    cluster_features = st.multiselect("Pilih fitur untuk clustering:", num_cols)
    n_clusters = st.slider("Jumlah cluster:", 2, 10, 3)
    if cluster_features:
        X_scaled = StandardScaler().fit_transform(df[cluster_features])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df['Cluster'] = clusters
        st.write("Cluster labels ditambahkan ke dataframe")

        if X_scaled.shape[1] >= 2:
            fig4, ax4 = plt.subplots()
            sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette="tab10", ax=ax4)
            ax4.set_title("Visualisasi Clustering")
            st.pyplot(fig4)
            st.markdown("""
            **Cara Baca Visualisasi Clustering:**
            - Tiap warna menunjukkan kelompok data yang serupa (cluster)
            - Gunakan untuk segmentasi pelanggan atau pengelompokan pola
            """)
        else:
            st.warning("Pilih minimal 2 fitur untuk visualisasi clustering.")

    st.subheader("📥 Export Prediksi dan Hasil")
    if 'Cluster' in df.columns:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="hasil_analisis.csv">Download hasil sebagai CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    st.subheader("📌 Outlier Detection (Z-Score)")
    outlier_feature = st.selectbox("Pilih fitur untuk deteksi outlier:", num_cols)
    if outlier_feature:
        z_scores = (df[outlier_feature] - df[outlier_feature].mean()) / df[outlier_feature].std()
        outliers = df[np.abs(z_scores) > 3]
        st.write(f"Jumlah outlier di kolom {outlier_feature}: {outliers.shape[0]}")
        st.dataframe(outliers)
        st.markdown("""
        **Penjelasan:**
        - Z-score di atas 3 atau di bawah -3 dianggap sebagai outlier (data menyimpang ekstrem dari rata-rata)
        """)

    st.subheader("📋 Ringkasan Otomatis (Natural Language)")
    st.markdown(f"""
    - Dataset memiliki **{df.shape[0]} baris** dan **{df.shape[1]} kolom**.
    - Fitur numerik utama: {', '.join(num_cols)}
    - Korelasi tertinggi antara fitur: `{df[num_cols].corr().abs().unstack().sort_values(ascending=False).drop_duplicates()[1:].idxmax()}`
    """)