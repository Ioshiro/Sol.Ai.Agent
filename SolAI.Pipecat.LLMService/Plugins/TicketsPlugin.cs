using System.ComponentModel;
using System.Text.Json;
using SolAI.Pipecat.LLMService.Data;
using SolAI.Pipecat.LLMService.Infrastructure;
using Microsoft.EntityFrameworkCore;
using Microsoft.SemanticKernel;

namespace SolAI.Pipecat.LLMService.Plugins;

public sealed class TicketsPlugin
{
    private readonly IDbContextFactory<TicketsDbContext> _dbContextFactory;

    public TicketsPlugin(IDbContextFactory<TicketsDbContext> dbContextFactory)
    {
        _dbContextFactory = dbContextFactory;
    }

    [KernelFunction("ottieni_tickets_aperti")]
    [Description("Restituisce i ticket aperti")]
    public string OttieniTicketsAperti()
    {
        return ToolExecutionLogger.Execute("Tickets", "ottieni_tickets_aperti", () =>
        {
            using var dbContext = _dbContextFactory.CreateDbContext();

            var aperti = dbContext.Tickets
                .AsNoTracking()
                .Where(t => t.Stato == Ticket.STATO_APERTO)
                .OrderByDescending(t => t.DataCreazione)
                .ToList();

            return JsonSerializer.Serialize(aperti, new JsonSerializerOptions { WriteIndented = true });
        });
    }

    [KernelFunction("ottieni_tickets_chiusi")]
    [Description("Restituisce i ticket chiusi nell'ultimo mese")]
    public string OttieniTicketsChiusi()
    {
        return ToolExecutionLogger.Execute("Tickets", "ottieni_tickets_chiusi", () =>
        {
            using var dbContext = _dbContextFactory.CreateDbContext();

            var dataChiusura = DateTime.Now.AddMonths(-1);
            var chiusi = dbContext.Tickets
                .AsNoTracking()
                .Where(t => t.Stato == Ticket.STATO_CHIUSO && t.DataChiusura >= dataChiusura)
                .OrderByDescending(t => t.DataChiusura)
                .ToList();

            return JsonSerializer.Serialize(chiusi, new JsonSerializerOptions { WriteIndented = true });
        });
    }

    [KernelFunction("apri_nuovo_ticket")]
    [Description("Apri un nuovo ticket")]
    public string ApriNuovoTicket(string titolo, string descrizione)
    {
        return ToolExecutionLogger.Execute("Tickets", "apri_nuovo_ticket", () =>
        {
            using var dbContext = _dbContextFactory.CreateDbContext();

            var ticket = new Ticket
            {
                Titolo = titolo,
                Descrizione = descrizione,
                DataCreazione = DateTime.Now,
                Stato = Ticket.STATO_APERTO,
                UtenteCreazione = "ai_assistant"
            };

            dbContext.Tickets.Add(ticket);
            dbContext.SaveChanges();

            return JsonSerializer.Serialize(ticket, new JsonSerializerOptions { WriteIndented = true });
        });
    }

    [KernelFunction("aggiungi_nota_ticket")]
    [Description("Aggiunge una nota a un ticket")]
    public string AggiungiNotaTicket(string ticketId, string nota)
    {
        return ToolExecutionLogger.Execute("Tickets", "aggiungi_nota_ticket", () =>
        {
            using var dbContext = _dbContextFactory.CreateDbContext();

            if (string.IsNullOrWhiteSpace(ticketId))
            {
                return "Ticket ID non valido";
            }

            if (!int.TryParse(ticketId, out var id))
            {
                return "Ticket ID non valido";
            }

            var ticket = dbContext.Tickets.Find(id);
            if (ticket is null)
            {
                return "Ticket non trovato";
            }

            if (string.IsNullOrWhiteSpace(nota))
            {
                return "Nota non valida";
            }

            ticket.Note = string.IsNullOrWhiteSpace(ticket.Note)
                ? nota
                : ticket.Note + "\n" + nota;

            dbContext.SaveChanges();

            return JsonSerializer.Serialize(ticket, new JsonSerializerOptions { WriteIndented = true });
        });
    }

    [KernelFunction("chiudi_ticket")]
    [Description("Chiude un ticket. Richiede sempre note di chiusura esplicite fornite dall'utente; se le note non sono state raccolte, il tool non deve essere chiamato.")]
    public string ChiudiTicket(string ticketId, string notaChiusura)
    {
        return ToolExecutionLogger.Execute("Tickets", "chiudi_ticket", () =>
        {
            using var dbContext = _dbContextFactory.CreateDbContext();

            if (string.IsNullOrWhiteSpace(ticketId))
            {
                return "Ticket ID non valido";
            }

            if (!int.TryParse(ticketId, out var id))
            {
                return "Ticket ID non valido";
            }

            if (string.IsNullOrWhiteSpace(notaChiusura))
            {
                return "Nota di chiusura non valida";
            }

            var ticket = dbContext.Tickets.Find(id);
            if (ticket is null)
            {
                return "Ticket non trovato";
            }

            ticket.Stato = Ticket.STATO_CHIUSO;
            ticket.DataChiusura = DateTime.Now;
            ticket.UtenteChiusura = "ai_assistant";
            ticket.Note = string.IsNullOrWhiteSpace(ticket.Note)
                ? "Nota di chiusura: " + notaChiusura
                : ticket.Note + "\nNota di chiusura: " + notaChiusura;

            dbContext.SaveChanges();

            return JsonSerializer.Serialize(ticket, new JsonSerializerOptions { WriteIndented = true });
        });
    }
}
