# whisper-standalone-win

Standalone executables of [OpenAI's Whisper](https://github.com/openai/whisper) & [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for those who don't want to bother with Python.

Executables are compatible with Windows 7 x64 and above.    
Meant to be used in command-line interface or [Subtitle Edit](https://github.com/SubtitleEdit/subtitleedit).   
Faster-Whisper is much faster than OpenAI's Whisper, and it requires less RAM/VRAM.

## Usage examples
* `whisper.exe "D:\videofile.mkv" --language en --model "medium"`   

* `whisper.exe --help`

## Notes

Run your command-line interface as Administrator.   
By default the subtitles are created in the same folder where an executable file is located.   
Programs automatically will choose to work on GPU if CUDA is detected.   
For decent transcription use not smaller than `small` model, `medium` is recommended.
   
## OpenAI standalone info

OpenAI version needs 'FFmpeg.exe' in PATH, or copy it to Whisper's folder [Subtitle Edit downloads FFmpeg automatically].
   
   
## Faster-Whisper standalone info

By default it looks for models in the same folder -> `_models\faster-whisper-medium`.   
Models are downloaded automatically or can be downloaded manually from: https://huggingface.co/guillaumekln   
In Subtitle Edit it can be selected for CTranslate2 engine, just rename it to `whisper-ctranslate2.exe`.   
"large" is mapped to `large-v2` model.
