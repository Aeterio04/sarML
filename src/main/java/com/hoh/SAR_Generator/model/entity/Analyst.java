package com.hoh.SAR_Generator.model.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Column;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "analyst")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Analyst {

    @Id
    private UUID id;

    @Column(nullable = false, unique = true)
    private String username;

    @Column(name = "password_hash", nullable = false)
    private String passwordHash;

    @Column(nullable = false)
    private String role;

    @Column(name = "created_at")
    private Instant createdAt;

    @Column(nullable = false)
    private String department;

    @Column(name = "last_login")
    private Instant lastLogin;
}