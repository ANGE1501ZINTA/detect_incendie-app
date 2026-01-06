"""
Projet: Détection et Classification des Incendies avec Réduction de Dimensionnalité Deep Learning
Groupe 9
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50, MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Configuration globale
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LATENT_DIM = 128  # Dimension de l'espace latent après réduction

class FireDatasetProcessor:
    """Préparation et prétraitement du dataset"""
    
    def __init__(self, data_path, img_size=IMG_SIZE):
        self.data_path = Path(data_path)
        self.img_size = img_size
        
    def load_and_split_data(self, test_size=0.2, val_size=0.1):
        """Charger et diviser le dataset"""
        images = []
        labels = []
        
        # Charger les images (adapter selon la structure du dataset)
        for label_dir in self.data_path.iterdir():
            if label_dir.is_dir():
                label = 1 if 'fire' in label_dir.name.lower() else 0
                for img_path in label_dir.glob('*.jpg'):
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    images.append(img)
                    labels.append(label)
        
        images = np.array(images) / 255.0  # Normalisation
        labels = np.array(labels)
        
        # Split train/temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=test_size+val_size, stratify=labels, random_state=42
        )
        
        # Split temp en validation/test
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_ratio), stratify=y_temp, random_state=42
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Distribution train - Fire: {sum(y_train)}, No Fire: {len(y_train)-sum(y_train)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_augmentation_generator(self):
        """Créer un générateur d'augmentation de données"""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )


class EmbeddingExtractor:
    """Extraction des embeddings avec CNN pré-entraînés"""
    
    def __init__(self, model_name='efficientnet', img_size=IMG_SIZE):
        self.model_name = model_name
        self.img_size = img_size
        self.model = self._build_model()
        
    def _build_model(self):
        """Construire le modèle d'extraction"""
        base_models = {
            'efficientnet': EfficientNetB0(include_top=False, pooling='avg', input_shape=(*self.img_size, 3)),
            'resnet50': ResNet50(include_top=False, pooling='avg', input_shape=(*self.img_size, 3)),
            'mobilenet': MobileNetV3Small(include_top=False, pooling='avg', input_shape=(*self.img_size, 3))
        }
        
        model = base_models.get(self.model_name, base_models['efficientnet'])
        model.trainable = False  # Freeze le modèle pré-entraîné
        return model
    
    def extract_embeddings(self, images):
        """Extraire les embeddings"""
        return self.model.predict(images, verbose=0)


class ConvolutionalAutoencoder:
    """Autoencodeur convolutionnel pour réduction de dimensionnalité"""
    
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        
    def build(self):
        """Construire l'autoencodeur"""
        # Encoder
        encoder_input = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(512, activation='relu')(encoder_input)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        encoded = layers.Dense(self.latent_dim, activation='relu', name='encoded')(x)
        
        self.encoder = models.Model(encoder_input, encoded, name='encoder')
        
        # Decoder
        decoder_input = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(256, activation='relu')(decoder_input)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(512, activation='relu')(x)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        self.decoder = models.Model(decoder_input, decoded, name='decoder')
        
        # Autoencoder complet
        autoencoder_output = self.decoder(self.encoder(encoder_input))
        self.autoencoder = models.Model(encoder_input, autoencoder_output, name='autoencoder')
        
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        return self.autoencoder
    
    def train(self, X_train, X_val, epochs=50):
        """Entraîner l'autoencodeur"""
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.autoencoder.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    
    def encode(self, X):
        """Encoder les données vers l'espace latent"""
        return self.encoder.predict(X, verbose=0)


class VariationalAutoencoder:
    """VAE pour réduction de dimensionnalité avec distribution latente"""
    
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.vae = None
        self.encoder = None
        
    def sampling(self, args):
        """Reparameterization trick"""
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def build(self):
        """Construire le VAE"""
        # Encoder
        encoder_input = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(512, activation='relu')(encoder_input)
        x = layers.Dense(256, activation='relu')(x)
        
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = layers.Lambda(self.sampling, name='z')([z_mean, z_log_var])
        
        self.encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        decoder_input = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(256, activation='relu')(decoder_input)
        x = layers.Dense(512, activation='relu')(x)
        decoder_output = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        decoder = models.Model(decoder_input, decoder_output, name='decoder')
        
        # VAE
        outputs = decoder(self.encoder(encoder_input)[2])
        self.vae = models.Model(encoder_input, outputs, name='vae')
        
        # Loss personnalisée
        reconstruction_loss = keras.losses.mse(encoder_input, outputs)
        reconstruction_loss *= self.input_dim
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        
        return self.vae
    
    def train(self, X_train, X_val, epochs=50):
        """Entraîner le VAE"""
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.vae.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    
    def encode(self, X):
        """Encoder vers l'espace latent"""
        z_mean, _, _ = self.encoder.predict(X, verbose=0)
        return z_mean


class EndToEndClassifier:
    """Classificateur end-to-end avec bottleneck intégré"""
    
    def __init__(self, base_model_name='efficientnet', latent_dim=LATENT_DIM):
        self.base_model_name = base_model_name
        self.latent_dim = latent_dim
        self.model = None
        
    def build(self):
        """Construire le modèle avec bottleneck"""
        # Base model
        if self.base_model_name == 'efficientnet':
            base = EfficientNetB0(include_top=False, pooling='avg', input_shape=(*IMG_SIZE, 3))
        elif self.base_model_name == 'resnet50':
            base = ResNet50(include_top=False, pooling='avg', input_shape=(*IMG_SIZE, 3))
        else:
            base = MobileNetV3Small(include_top=False, pooling='avg', input_shape=(*IMG_SIZE, 3))
        
        base.trainable = False
        
        # Classification head avec bottleneck
        inputs = layers.Input(shape=(*IMG_SIZE, 3))
        x = base(inputs)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Bottleneck layer
        bottleneck = layers.Dense(self.latent_dim, activation='relu', name='bottleneck')(x)
        x = layers.Dropout(0.3)(bottleneck)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs, outputs)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """Entraîner le modèle"""
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history
    
    def get_bottleneck_model(self):
        """Extraire le modèle jusqu'au bottleneck"""
        bottleneck_output = self.model.get_layer('bottleneck').output
        return models.Model(self.model.input, bottleneck_output)


class Evaluator:
    """Évaluation et visualisation des résultats"""
    
    @staticmethod
    def plot_training_history(history, title='Training History'):
        """Visualiser l'historique d'entraînement"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_title(f'{title} - Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy (si disponible)
        if 'accuracy' in history.history:
            axes[1].plot(history.history['accuracy'], label='Train Accuracy')
            axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
            axes[1].set_title(f'{title} - Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def evaluate_classifier(y_true, y_pred, title='Confusion Matrix'):
        """Évaluer et afficher les métriques"""
        print("\n" + "="*50)
        print(f"ÉVALUATION: {title}")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=['No Fire', 'Fire']))
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Fire', 'Fire'],
                    yticklabels=['No Fire', 'Fire'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
    
    @staticmethod
    def visualize_embeddings_2d(embeddings, labels, title='Embeddings Visualization'):
        """Visualiser les embeddings en 2D avec t-SNE"""
        from sklearn.manifold import TSNE
        
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='coolwarm', alpha=0.6)
        plt.colorbar(scatter, label='Fire (1) / No Fire (0)')
        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.show()


# Pipeline principal
def main_pipeline(data_path):
    """Pipeline complet du projet"""
    
    print("="*70)
    print("PROJET: DÉTECTION D'INCENDIES AVEC RÉDUCTION DE DIMENSIONNALITÉ")
    print("="*70)
    
    # 1. Préparation des données
    print("\n[1/6] Préparation des données...")
    processor = FireDatasetProcessor(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = processor.load_and_split_data()
    
    # 2. Extraction des embeddings
    print("\n[2/6] Extraction des embeddings CNN...")
    extractor = EmbeddingExtractor(model_name='efficientnet')
    train_embeddings = extractor.extract_embeddings(X_train)
    val_embeddings = extractor.extract_embeddings(X_val)
    test_embeddings = extractor.extract_embeddings(X_test)
    
    print(f"Dimension des embeddings: {train_embeddings.shape[1]}")
    
    # 3. Réduction de dimensionnalité avec CAE
    print("\n[3/6] Réduction de dimensionnalité (Autoencoder)...")
    cae = ConvolutionalAutoencoder(input_dim=train_embeddings.shape[1], latent_dim=LATENT_DIM)
    cae.build()
    cae_history = cae.train(train_embeddings, val_embeddings, epochs=EPOCHS)
    
    train_reduced_cae = cae.encode(train_embeddings)
    test_reduced_cae = cae.encode(test_embeddings)
    
    # 4. Classification avec SVM sur embeddings réduits
    print("\n[4/6] Classification (SVM sur embeddings réduits)...")
    svm_reduced = SVC(kernel='rbf', probability=True, random_state=42)
    svm_reduced.fit(train_reduced_cae, y_train)
    y_pred_reduced = svm_reduced.predict(test_reduced_cae)
    
    # 5. Classification baseline (sans réduction)
    print("\n[5/6] Classification baseline (SVM sur embeddings originaux)...")
    svm_baseline = SVC(kernel='rbf', probability=True, random_state=42)
    svm_baseline.fit(train_embeddings, y_train)
    y_pred_baseline = svm_baseline.predict(test_embeddings)
    
    # 6. Évaluation
    print("\n[6/6] Évaluation et visualisation...")
    evaluator = Evaluator()
    
    evaluator.plot_training_history(cae_history, 'Autoencoder Training')
    
    metrics_reduced = evaluator.evaluate_classifier(y_test, y_pred_reduced, 
                                                     'Avec Réduction (CAE + SVM)')
    metrics_baseline = evaluator.evaluate_classifier(y_test, y_pred_baseline,
                                                      'Sans Réduction (Embeddings + SVM)')
    
    # Visualisation des embeddings
    evaluator.visualize_embeddings_2d(test_reduced_cae, y_test, 
                                      'Embeddings Réduits (CAE)')
    
    # Comparaison
    print("\n" + "="*70)
    print("COMPARAISON DES PERFORMANCES")
    print("="*70)
    print(f"Avec réduction (CAE):   Accuracy={metrics_reduced['accuracy']:.4f}, F1={metrics_reduced['f1_score']:.4f}")
    print(f"Sans réduction:         Accuracy={metrics_baseline['accuracy']:.4f}, F1={metrics_baseline['f1_score']:.4f}")
    print(f"Compression:            {train_embeddings.shape[1]} → {LATENT_DIM} dimensions")
    
    return {
        'cae': cae,
        'svm_reduced': svm_reduced,
        'svm_baseline': svm_baseline,
        'extractor': extractor
    }


if __name__ == "__main__":
    # Exemple d'utilisation
    DATA_PATH = "path/to/fire-recognition-dataset"  # À adapter
    
    # Exécuter le pipeline
    # models = main_pipeline(DATA_PATH)
    
    print("\n✓ Code prêt à être exécuté!")
    print("Remplacez DATA_PATH par le chemin vers votre dataset.")
