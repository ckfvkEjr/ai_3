import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '14y7xPjVyBg_oFasSuODSHQP3b2BSm6vt'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

# 멜 스펙트로그램 생성 함수 (이미지 저장)
def create_mel_spectrogram(audio_file, output_path="mel_spectrogram.png"):
    # 오디오 로드
    y, sr = librosa.load(audio_file)

    # 멜 스펙트로그램 생성
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 멜 스펙트로그램 시각화 및 저장
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return output_path

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_container_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_container_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    'disco': {
        'images': [
            "https://i.ibb.co/DYRXfhr/tenor.gif",
            "https://i.ibb.co/DYRXfhr/tenor.gif",
            "https://i.ibb.co/DYRXfhr/tenor.gif"
        ],
        'videos': [
            "https://youtu.be/I_izvAbhExY?si=HxpKfg098BfLDtyW",
            "https://youtu.be/yURRmWtbTbo?si=Utom01m0eX3P6sZK",
            "https://youtu.be/CS9OO0S5w2k?si=v7n75cjv6iDEpLho"
        ],
        'texts': [
            "디스코~~",
            "디스코~~",
            "디스코~~"
        ]
    },
    'blues': {
        'images': [
            "https://i.ibb.co/7YgGLdN/image.jpg",
            "https://i.ibb.co/7YgGLdN/image.jpg",
            "https://i.ibb.co/7YgGLdN/image.jpg"
        ],
        'videos': [
            "https://youtu.be/4zAThXFOy2c?si=6yL37t7qeRF6gNvd",
            "https://youtu.be/71Gt46aX9Z4?si=9AxWcxGKutLfQKUB",
            "https://youtu.be/SgXSomPE_FY?si=01EMEwoh6aRKPvDT"
        ],
        'texts': [
            "블루스~~",
            "블루스~~",
            "블루스~~"
        ]
    },
    'classical': {
        'images': [
            "https://i.ibb.co/3k7wmkZ/14668.webp",
            "https://i.ibb.co/3k7wmkZ/14668.webp",
            "https://i.ibb.co/3k7wmkZ/14668.webp"
        ],
        'videos': [
            "https://youtu.be/l0GN40EL1VU?si=ueYMu6WwoDe9WwSO",
            "https://youtu.be/4exkCrFCBps?si=j1JyYf35P66Rz08O",
            "https://youtu.be/p29JUpsOSTE?si=xBBtvgALxiADuZHE"
        ],
        'texts': [
            "클래식~~",
            "클래식~~",
            "클래식~~"
        ]
    },
    'country': {
        'images': [
            "https://i.ibb.co/WVJP2hL/funny-country-music-memes-2.jpg",
            "https://i.ibb.co/WVJP2hL/funny-country-music-memes-2.jpg",
            "https://i.ibb.co/WVJP2hL/funny-country-music-memes-2.jpg"
        ],
        'videos': [
            "https://youtu.be/7qaHdHpSjX8?si=SkmYtU-ow6kmTBZ6",
            "https://youtu.be/dRX0wDNK6S4?si=Vl40DjWjQTv6HwWz",
            "https://youtu.be/WBDpb7SwSgU?si=7g3el6N5VHOhVjGW"
        ],
        'texts': [
            "컨트리~~",
            "컨트리~~",
            "컨트리~~"
        ]
    },
    'hiphop': {
        'images': [
            "https://i.ibb.co/sJNvxwV/images.jpg",
            "https://i.ibb.co/sJNvxwV/images.jpg",
            "https://i.ibb.co/sJNvxwV/images.jpg"
        ],
        'videos': [
            "https://youtu.be/DmWWqogr_r8?si=XOHGRx5O7SQEsxxS",
            "https://youtu.be/S9bCLPwzSC0?si=vHEK7ZjkfeaHXOI6",
            "https://youtu.be/tvTRZJ-4EyI?si=dgQQh2G6RYwu5n4V"
        ],
        'texts': [
            "힙합~~",
            "힙합~~",
            "힙합~~"
        ]
    },
    'jazz': {
        'images': [
            "https://i.ibb.co/w6K3sNw/10d723cdc89427f05a38f350103ed792.jpg",
            "https://i.ibb.co/w6K3sNw/10d723cdc89427f05a38f350103ed792.jpg",
            "https://i.ibb.co/w6K3sNw/10d723cdc89427f05a38f350103ed792.jpg"
        ],
        'videos': [
            "https://youtu.be/Cv9NSR-2DwM?si=nx66HKo5SoH9CakP",
            "https://youtu.be/ylXk1LBvIqU?si=-PPM6GP9wcUG_ZMN",
            "https://youtu.be/-488UORrfJ0?si=UZVnsAVC4N-kaSz9"
        ],
        'texts': [
            "재즈~~",
            "재즈~~",
            "재즈~~"
        ]
    }
}


# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 음악 파일 업로드
uploaded_music = st.file_uploader("음악 파일을 업로드하세요 (MP3, WAV)", type=['mp3', 'wav'])


if uploaded_music is not None:
     # 멜 스펙트로그램 생성 및 저장
    mel_spec_path = create_mel_spectrogram(uploaded_music)

    # 이미지 열기
    mel_spec_image = PILImage.create(mel_spec_path)

    prediction, _, probs = learner.predict(mel_spec_path)
    

    with left_column:
        display_left_content(mel_spec_path, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)
