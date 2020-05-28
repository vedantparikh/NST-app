mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
eanbleCORS = false\n\
\n\
" > ~/.streamlit/config.toml