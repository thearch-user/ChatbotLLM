-- ChatbotLLM Neovim Plugin
-- Save this file to ~/.config/nvim/lua/chatbotllm.lua
-- Or in your project: extensions/neovim/chatbotllm.lua

local M = {}

function M.complete()
    local bufnr = vim.api.nvim_get_current_buf()
    local start_line, start_col = unpack(vim.api.nvim_win_get_cursor(0))
    
    -- Get current buffer text
    local lines = vim.api.nvim_buf_get_lines(bufnr, 0, -1, false)
    local text = table.concat(lines, "\n")
    
    print("ChatbotLLM: Thinking...")
    
    -- Use curl to call the backend
    local cmd = string.format(
        "curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{\"text\": %s}'",
        vim.fn.shellescape(text)
    )
    
    local handle = io.popen(cmd)
    local result = handle:read("*a")
    handle:close()
    
    -- Parse JSON (simple parsing or use a library like vim.fn.json_decode)
    local ok, decoded = pcall(vim.fn.json_decode, result)
    if ok and decoded.prediction then
        local prediction = decoded.prediction
        -- Insert prediction at current cursor
        local prediction_lines = vim.split(prediction, "\n")
        vim.api.nvim_buf_set_lines(bufnr, start_line, start_line, false, prediction_lines)
        print("ChatbotLLM: Done")
    else
        print("ChatbotLLM: Error calling backend")
    end
end

-- Keybinding example:
-- vim.keymap.set('n', '<leader>cc', require('chatbotllm').complete, { desc = 'ChatbotLLM Complete' })

return M
