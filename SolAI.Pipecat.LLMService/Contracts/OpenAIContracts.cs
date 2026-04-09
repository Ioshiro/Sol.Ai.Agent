using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SolAI.Pipecat.LLMService.Contracts;

public sealed class OpenAIChatCompletionRequest
{
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("messages")]
    public List<OpenAIChatMessage> Messages { get; init; } = [];

    [JsonPropertyName("stream")]
    public bool Stream { get; init; }

    [JsonPropertyName("temperature")]
    public double? Temperature { get; init; }

    [JsonPropertyName("top_p")]
    public double? TopP { get; init; }

    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; init; }

    [JsonPropertyName("user")]
    public string? User { get; init; }

    [JsonPropertyName("tools")]
    public JsonElement Tools { get; init; }

    [JsonPropertyName("tool_choice")]
    public JsonElement ToolChoice { get; init; }
}

public sealed class OpenAIChatMessage
{
    [JsonPropertyName("role")]
    public string Role { get; init; } = string.Empty;

    [JsonPropertyName("content")]
    public JsonElement Content { get; init; }

    [JsonPropertyName("name")]
    public string? Name { get; init; }

    [JsonPropertyName("tool_call_id")]
    public string? ToolCallId { get; init; }

    public string ExtractTextContent()
    {
        return ExtractText(Content);
    }

    private static string ExtractText(JsonElement content)
    {
        return content.ValueKind switch
        {
            JsonValueKind.String => content.GetString() ?? string.Empty,
            JsonValueKind.Object => ExtractTextFromObject(content),
            JsonValueKind.Array => ExtractTextFromArray(content),
            JsonValueKind.Undefined or JsonValueKind.Null => string.Empty,
            _ => content.GetRawText()
        };
    }

    private static string ExtractTextFromObject(JsonElement content)
    {
        if (content.TryGetProperty("text", out var textProperty) &&
            textProperty.ValueKind == JsonValueKind.String)
        {
            return textProperty.GetString() ?? string.Empty;
        }

        return content.GetRawText();
    }

    private static string ExtractTextFromArray(JsonElement content)
    {
        var builder = new StringBuilder();

        foreach (var item in content.EnumerateArray())
        {
            if (item.ValueKind == JsonValueKind.String)
            {
                builder.AppendLine(item.GetString());
                continue;
            }

            if (item.ValueKind != JsonValueKind.Object)
            {
                continue;
            }

            if (item.TryGetProperty("type", out var typeProperty) &&
                typeProperty.ValueKind == JsonValueKind.String &&
                !string.Equals(typeProperty.GetString(), "text", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (item.TryGetProperty("text", out var textProperty) &&
                textProperty.ValueKind == JsonValueKind.String)
            {
                builder.AppendLine(textProperty.GetString());
            }
        }

        return builder.ToString().Trim();
    }
}

public sealed class OpenAIChatCompletionResponse
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = string.Empty;

    [JsonPropertyName("object")]
    public string Object { get; init; } = "chat.completion";

    [JsonPropertyName("created")]
    public long Created { get; init; }

    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    [JsonPropertyName("choices")]
    public List<OpenAIChatChoice> Choices { get; init; } = [];
}

public sealed class OpenAIChatChoice
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("message")]
    public OpenAIAssistantMessage Message { get; init; } = new();

    [JsonPropertyName("finish_reason")]
    public string FinishReason { get; init; } = "stop";
}

public sealed class OpenAIAssistantMessage
{
    [JsonPropertyName("role")]
    public string Role { get; init; } = "assistant";

    [JsonPropertyName("content")]
    public string Content { get; init; } = string.Empty;
}

public sealed class OpenAIChatCompletionChunk
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = string.Empty;

    [JsonPropertyName("object")]
    public string Object { get; init; } = "chat.completion.chunk";

    [JsonPropertyName("created")]
    public long Created { get; init; }

    [JsonPropertyName("model")]
    public string Model { get; init; } = string.Empty;

    [JsonPropertyName("choices")]
    public List<OpenAIChatChunkChoice> Choices { get; init; } = [];
}

public sealed class OpenAIChatChunkChoice
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("delta")]
    public OpenAIChatDelta Delta { get; init; } = new();

    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; init; }
}

public sealed class OpenAIChatDelta
{
    [JsonPropertyName("role")]
    public string? Role { get; init; }

    [JsonPropertyName("content")]
    public string? Content { get; init; }
}

public sealed class OpenAIModelListResponse
{
    [JsonPropertyName("object")]
    public string Object { get; init; } = "list";

    [JsonPropertyName("data")]
    public List<OpenAIModelCard> Data { get; init; } = [];
}

public sealed class OpenAIModelCard
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = string.Empty;

    [JsonPropertyName("object")]
    public string Object { get; init; } = "model";

    [JsonPropertyName("created")]
    public long Created { get; init; }

    [JsonPropertyName("owned_by")]
    public string OwnedBy { get; init; } = "solai-llm-service";
}
