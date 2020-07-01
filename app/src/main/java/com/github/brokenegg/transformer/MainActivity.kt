package com.github.brokenegg.transformer

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val t = Translator(this)
        t.run()
    }
}