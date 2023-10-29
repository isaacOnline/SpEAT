import os

import pydub


class ScrapingError(Exception):
    pass


def _convert_mp3_to_wav(file_name, file_dir):
    """
    Convert a single mp3 file to a wav file

    Read the mp3 file from disk, write a new wav file with an updated extension, and delete the original mp3 file

    :param file_name: Identification for the speaker whose audio should be converted. Should not include extension
    :param file_dir: Relative path from project root to file
    :return:
    """
    mp3_fp = os.path.join(file_dir, f'{file_name}.mp3')
    wav_fp = os.path.join(file_dir, f'{file_name}.wav')

    if not os.path.exists('data') or not os.path.exists(file_dir):
        raise ScrapingError('Data has not been downloaded')

    if os.path.exists(wav_fp):
        if os.path.exists(mp3_fp):
            os.remove(mp3_fp)
        speaker_audio = pydub.AudioSegment.from_wav(wav_fp)
        speaker_audio = speaker_audio.set_frame_rate(16000)
        speaker_audio.export(wav_fp, format='wav')
        return int(speaker_audio.frame_count())

    if not os.path.exists(mp3_fp):
        raise ScrapingError(f'MP3 for speaker {mp3_fp} does not have a downloaded mp3 or wav')

    speaker_audio = pydub.AudioSegment.from_mp3(mp3_fp)
    speaker_audio = speaker_audio.set_frame_rate(16000)
    speaker_audio.export(wav_fp, format='wav')
    os.remove(mp3_fp)
    return int(speaker_audio.frame_count())


def _convert_wav_to_wav(file_name, file_dir, condense_channels = False):
    """
    Make sure a wav file has the desired frame rate

    Returns the number of frames that are contained in the wav file

    :param file_name: Identification for the speaker whose audio should be converted. Should not include extension
    :param file_dir: Relative path from project root to file
    :param condense_channels: Whether file should be converted to mono
    :return:
    """
    wav_fp = os.path.join(file_dir, f'{file_name}.wav')

    if not os.path.exists(wav_fp):
        raise ScrapingError(f'Data has not been downloaded: {wav_fp}')

    speaker_audio = pydub.AudioSegment.from_wav(wav_fp)
    speaker_audio = speaker_audio.set_frame_rate(16000)
    if condense_channels:
        if speaker_audio.channels != 1:
            print(f'File {file_name} had {speaker_audio.channels} channels originally, but is being converted to a '
                  f'single channel')
        speaker_audio = speaker_audio.set_channels(1)
    else:
        print(f'File {file_name} has {speaker_audio.channels} channels')
    speaker_audio.export(wav_fp, format='wav')
    return int(speaker_audio.frame_count())
