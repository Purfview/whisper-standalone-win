[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate?hosted_button_id=JF5BEQE3YQGH2)   

# whisper-standalone-win

![alt text](https://i.imgur.com/DYVm3u6.png)

Standalone executables of [OpenAI's Whisper](https://github.com/openai/whisper) & [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for those who don't want to bother with Python.

Faster-Whisper executables are compatible with Windows 7 x64, Linux v5.4, macOS v10.15 and above.    
Meant to be used in command-line interface or in programs like [Subtitle Edit](https://github.com/SubtitleEdit/subtitleedit), [Tero Subtitler](https://github.com/URUWorks/TeroSubtitler), [FFAStrans](https://ffastrans.com/wp/).     
Faster-Whisper is much faster & better than OpenAI's Whisper, and it requires less RAM/VRAM.

## Usage examples:
* `whisper-faster.exe "D:\videofile.mkv" --language=English --model=medium`   

* `whisper-faster.exe --help`

## Notes:

Executables & libs can be downloaded from `Releases`. [at the right side of this page]    
Don't copy programs to the Windows' folders! [run as Administrator if you did]       
Programs automatically will choose to work on GPU if CUDA is detected.   
For decent transcription use not smaller than `medium` model.   
Guide how to run the command line programs: https://www.youtube.com/watch?v=A3nwRCV-bTU   
Examples how to do batch processing on the multiple files: https://github.com/Purfview/whisper-standalone-win/discussions/29   
   
   
## Faster-Whisper standalone info:

Some defaults are tweaked for movies transcriptions and to make it portable.   
Shows the progress bar in the title bar of command-line interface. [or it can be printed with `-pp`]   
By default it looks for models in the same folder, in path like this -> `_models\faster-whisper-medium`.   
Models are downloaded automatically or can be downloaded manually from: https://huggingface.co/Systran        
`beam_size=1`: can speed-up transcription twice. [ in my tests it had insignificant impact on accuracy ]     
`compute_type`: test different types to find fastest for your hardware. [`--verbose=true` to see all supported types]    
To reduce memory usage try `--best_of=1`, or `--temperature_increment_on_fallback=None`.   

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate?hosted_button_id=JF5BEQE3YQGH2)


