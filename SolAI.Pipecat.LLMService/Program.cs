using System.Diagnostics;
using System.Text.Json.Serialization;
using OpenTelemetry.Exporter;
using OpenTelemetry.Trace;
using SolAI.Pipecat.LLMService.Contracts;
using SolAI.Pipecat.LLMService.Options;
using SolAI.Pipecat.LLMService.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddProblemDetails();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
});

builder.Services.AddOpenTelemetry()
    .WithTracing(tracing =>
    {
        var langfuseHost = builder.Configuration["LANGFUSE_HOST"] ?? "http://localhost:3000";
        var publicKey = builder.Configuration["LANGFUSE_PUBLIC_KEY"] ?? "lf_pk_local_demo";
        var secretKey = builder.Configuration["LANGFUSE_SECRET_KEY"] ?? "lf_sk_local_demo";
        var auth = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes($"{publicKey}:{secretKey}"));

        tracing
            .AddAspNetCoreInstrumentation()
            .AddHttpClientInstrumentation()
            .AddSource(SemanticKernelOpenAIChatGateway.ActivitySourceName)
            .AddOtlpExporter(options =>
            {
                options.Endpoint = new Uri($"{langfuseHost.TrimEnd('/')}/api/public/otel");
                options.Protocol = OtlpExportProtocol.HttpProtobuf;
                options.Headers = $"Authorization=Basic {auth}";
            });
    });

builder.Services
    .AddOptions<LlmServiceOptions>()
    .Bind(builder.Configuration.GetSection(LlmServiceOptions.SectionName))
    .ValidateDataAnnotations()
    .ValidateOnStart();

builder.Services.AddSingleton<IOpenAIChatGateway, SemanticKernelOpenAIChatGateway>();
builder.Services.AddHealthChecks();

var app = builder.Build();

app.UseExceptionHandler();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.MapHealthChecks("/health");

app.MapGet("/", () => Results.Redirect("/swagger"));

app.MapGet("/v1/models", (IOpenAIChatGateway gateway) =>
{
    Activity.Current?.SetTag("langfuse.trace.name", "list-models");

    var models = gateway.GetModels();
    Activity.Current?.SetTag("llm.models.count", models.Data.Count);

    return Results.Ok(models);
})
.WithName("ListModels");

app.MapPost("/v1/chat/completions", async (
    OpenAIChatCompletionRequest request,
    IOpenAIChatGateway gateway,
    HttpContext httpContext,
    CancellationToken cancellationToken) =>
{
    Activity.Current?.SetTag("langfuse.trace.name", "chat-completions");
    Activity.Current?.SetTag("llm.request.stream", request.Stream);
    Activity.Current?.SetTag("llm.request.message_count", request.Messages.Count);
    Activity.Current?.SetTag("llm.request.model", request.Model ?? string.Empty);

    if (request.Messages.Count == 0)
    {
        Activity.Current?.SetTag("llm.request.valid", false);
        return Results.ValidationProblem(new Dictionary<string, string[]>
        {
            ["messages"] = ["E' richiesto almeno un messaggio."]
        });
    }

    Activity.Current?.SetTag("llm.request.valid", true);

    if (request.Stream)
    {
        Activity.Current?.SetTag("llm.request.mode", "stream");
        await gateway.WriteStreamingCompletionAsync(request, httpContext.Response, cancellationToken);
        return Results.Empty;
    }

    Activity.Current?.SetTag("llm.request.mode", "non_stream");
    var response = await gateway.CreateCompletionAsync(request, cancellationToken);
    Activity.Current?.SetTag("llm.response.model", response.Model);
    Activity.Current?.SetTag("llm.response.choice_count", response.Choices.Count);

    return Results.Ok(response);
})
.WithName("CreateChatCompletion");

app.Run();
