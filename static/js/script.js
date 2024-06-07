document.addEventListener('DOMContentLoaded', (event) => {
    const openButton = document.getElementById('open-popup-button');
    const closeButton = document.getElementById('close-popup-button');
    const popup = document.getElementById('popup');
    const overlay = document.getElementById('overlay');

    openButton.addEventListener('click', () => {
        popup.style.display = 'block';
        overlay.style.display = 'block';
    });

    closeButton.addEventListener('click', () => {
        popup.style.display = 'none';
        overlay.style.display = 'none';
    });

    overlay.addEventListener('click', () => {
        popup.style.display = 'none';
        overlay.style.display = 'none';
    });
});
