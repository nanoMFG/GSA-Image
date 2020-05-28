
export PATH="$(brew --prefix tcl-tk)/bin:$PATH" 
export LDFLAGS="-L$(brew --prefix tcl-tk)/lib" 
export CPPFLAGS="-I$(brew --prefix tcl-tk)/include" 
export PKG_CONFIG_PATH="$(brew --prefix tcl-tk)/lib/pkgconfig" 
export CFLAGS="-I$(brew --prefix tcl-tk)/include" 
export PYTHON_CONFIGURE_OPTS="--with-tcltk-includes='-I$(brew --prefix tcl-tk)/include' --with-tcltk-libs='-L$(brew --prefix tcl-tk)/lib -ltcl8.6 -ltk8.6'" 
