using Microsoft.EntityFrameworkCore;

namespace SolAI.Pipecat.LLMService.Data;

public sealed class TicketsDbContext : DbContext
{
    public TicketsDbContext(DbContextOptions<TicketsDbContext> options)
        : base(options)
    {
    }

    public DbSet<Ticket> Tickets => Set<Ticket>();
}
