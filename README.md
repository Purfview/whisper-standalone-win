[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate?hosted_button_id=JF5BEQE3YQGH2)   

# whisper-standalone-win

![alt text](https://i.imgur.com/DYVm3u6.png)

Standalone executables of [OpenAI's Whisper](https://github.com/openai/whisper) & [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for those who don't want to bother with Python.

Executables are compatible with Windows 7 x64 and above.    
Meant to be used in command-line interface or [Subtitle Edit](https://github.com/SubtitleEdit/subtitleedit).   
Faster-Whisper is much faster than OpenAI's Whisper, and it requires less RAM/VRAM.

## Usage examples:
* `whisper-faster.exe "D:\videofile.mkv" --language=English --model=medium`   

* `whisper-faster.exe --help`

## Notes:

Run your command-line interface as Administrator.   
Don't copy programs to the Windows' folders!    
Programs automatically will choose to work on GPU if CUDA is detected.   
For decent transcription use not smaller than `medium` model.   
Guide how to run the command line programs: https://www.youtube.com/watch?v=A3nwRCV-bTU   
Examples how to do batch processing on the multiple files: https://github.com/Purfview/whisper-standalone-win/discussions/29   
   
## OpenAI's Whisper standalone info:
   
By default the subtitles are created in the current folder.   
Needs 'FFmpeg.exe' in PATH, or copy it to Whisper's folder [Subtitle Edit downloads FFmpeg automatically].
   
   
## Faster-Whisper standalone info:

Some defaults are tweaked for movies transcriptions and to make it portable.   
Shows the progress bar in the title bar of command-line interface.   
By default it looks for models in the same folder, in path like this -> `_models\faster-whisper-medium`.   
Models are downloaded automatically or can be downloaded manually from: https://huggingface.co/guillaumekln   
In Subtitle Edit it can be selected for CTranslate2 engine, just rename it to `whisper-ctranslate2.exe`.   
By default the subtitles are created in the same folder where the executable file is located.   
`large` is mapped to `large-v2` model.   
`beam_size=5` potentially can improve transcription or make some halucination go away, but makes it ~twice slower.      

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate?hosted_button_id=JF5BEQE3YQGH2)


